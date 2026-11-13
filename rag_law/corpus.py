# -*- coding: utf-8 -*-
"""Corpus access: UTS_VLC 2026 split + vn_law.sqlite catalogue.

The sqlite `laws` table is the authoritative source for temporal metadata
(issue_date, eff_from, eff_to, status) because 298/306 corpus docs do NOT
carry an in-document effective-date header.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from rag_law.utils import extract_doc_refs, norm_doc_num

DEFAULT_ROOT = Path(r"D:\GiangTV\RAG_LAW_Vnew\data\uts_vlc")

# Docs whose body in the 2026 split is known to be wrong/invalid (Phase 0
# data validation). See RAG_PLAN.md §2.4 — 45/2019/QH14 (Bộ luật Lao động)
# contains a provincial plan document instead of the Labor Code.
DATA_ERROR_DOCS: dict[str, str] = {
    "45/2019/QH14": "body sai nội dung (thực thể là Kế hoạch 292/KH-UBND tỉnh Lào Cai); chờ thay bằng toàn văn Bộ luật Lao động 2019",
}


@dataclass
class DocMeta:
    doc_num: str
    title: str
    doc_type: str = "law"          # constitution | code | law
    year: str | None = None
    issue_date: str | None = None
    eff_from: str | None = None
    eff_to: str | None = None
    status_name: str | None = None
    in_force: bool = True
    in_corpus: bool = False        # full text present in the 2026 split
    url_vbpl: str | None = None
    data_error: str | None = None  # reason if body is unreliable
    header_eff_date: str | None = None


class Corpus:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else DEFAULT_ROOT
        self.docs: dict[str, DocMeta] = {}       # doc_num -> meta (in-corpus only)
        self.contents: dict[str, str] = {}       # doc_num -> full markdown text
        self.catalog: dict[str, DocMeta] = {}    # doc_num -> meta (full catalogue, incl. non-corpus)
        self.missing: set[str] = set()           # in vbpl catalogue, absent from corpus
        self.amends: dict[str, list[str]] = {}   # doc_num -> [doc_nums it amends]
        self.amended_by: dict[str, list[str]] = {}
        self.replaces: dict[str, list[str]] = {}  # lapsed doc -> newer same-title doc(s) in force
        self.snapshot_date = "2026-06"
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        self._load_parquet()
        self._load_sqlite()
        self._build_amendments()
        self._build_replacements()

    def _load_parquet(self) -> None:
        path = next(self.root.glob("data/2026-*.parquet"))
        for d in pq.read_table(path).to_pylist():
            dn = norm_doc_num(d["id"])
            meta = DocMeta(
                doc_num=dn,
                title=d["title"],
                doc_type=d["type"],
                in_corpus=True,
            )
            if dn in DATA_ERROR_DOCS:
                meta.data_error = DATA_ERROR_DOCS[dn]
            hm = re.search(r"\*\*Ngày hiệu lực:\*\*\s*([^\n|]+)", d["content"])
            if hm:
                meta.header_eff_date = hm.group(1).strip()
            self.docs[dn] = meta
            self.contents[dn] = d["content"]
            self.catalog[dn] = meta

    def _load_sqlite(self) -> None:
        con = sqlite3.connect(str(self.root / "database" / "vn_law.sqlite"))
        con.row_factory = sqlite3.Row
        try:
            for r in con.execute(
                "SELECT doc_num, title, doc_type, year, issue_date, eff_from,"
                " eff_to, status_name, in_force FROM laws"
            ):
                dn = norm_doc_num(r["doc_num"])
                meta = self.catalog.get(dn) or DocMeta(doc_num=dn, title=r["title"] or "")
                meta.title = meta.title or (r["title"] or "")
                meta.year = r["year"]
                meta.issue_date = r["issue_date"]
                meta.eff_from = r["eff_from"]
                meta.eff_to = r["eff_to"]
                meta.status_name = r["status_name"]
                meta.in_force = bool(r["in_force"])
                if dn not in self.docs:
                    meta.in_corpus = False
                self.catalog[dn] = meta
            for r in con.execute("SELECT doc_num, url_vbpl FROM law_links"):
                dn = norm_doc_num(r["doc_num"])
                if dn in self.catalog and r["url_vbpl"]:
                    self.catalog[dn].url_vbpl = r["url_vbpl"]
            self.missing = {
                norm_doc_num(r["doc_num"])
                for r in con.execute("SELECT doc_num FROM missing_in_vlc")
            }
        finally:
            con.close()
        # Constitution is not keyed by number in sqlite.
        if "HIẾN PHÁP 2013" not in self.catalog and "Hiến pháp 2013" not in self.catalog:
            self.catalog["HIẾN PHÁP 2013"] = DocMeta(
                doc_num="HIẾN PHÁP 2013",
                title="Hiến pháp năm 2013",
                doc_type="constitution",
                in_force=True,
                status_name="Còn hiệu lực",
                in_corpus="HIẾN PHÁP 2013" in self.docs,
            )

    def _build_amendments(self) -> None:
        """Extract amends relations from law titles ('Luật sửa đổi ... của Luật X số NN/..')."""
        for dn, meta in self.docs.items():
            refs = [r for r in extract_doc_refs(meta.title) if r != dn]
            if refs:
                self.amends[dn] = refs
                for base in refs:
                    self.amended_by.setdefault(base, [])
                    if dn not in self.amended_by[base]:
                        self.amended_by[base].append(dn)

    _TITLE_SUFFIX_RE = re.compile(r"\s*so\s+\d{1,3}/\d{4}/[a-z0-9\-]+")

    @classmethod
    def _base_title(cls, m: DocMeta) -> str:
        """Folded title without the 'số NNN/NNNN/TYPE' suffix, so
        'Luật Đất đai số 45/2013/QH13' and 'Luật Đất đai' group together."""
        from rag_law.utils import fold

        return cls._TITLE_SUFFIX_RE.sub("", fold(m.title or "")).strip()

    def _build_replacements(self) -> None:
        """Same base title + same doc_type + later year + in force → the
        lapsed document was very likely replaced by it (e.g. Luật Đất đai
        45/2013/QH13 → 31/2024/QH15)."""
        groups: dict[tuple[str, str], list[tuple[str, DocMeta]]] = {}
        for dn, m in self.catalog.items():
            if "/" not in dn or not m.title:
                continue
            groups.setdefault((self._base_title(m), m.doc_type), []).append((dn, m))
        for group in groups.values():
            group.sort(key=lambda x: (str(x[1].year or ""), x[0]))
            for i, (dn, m) in enumerate(group):
                later = [dn2 for dn2, m2 in group[i + 1:] if m2.in_force]
                if later and not m.in_force:
                    self.replaces[dn] = later

    def replaced_by(self, doc_num: str) -> list[str]:
        return self.replaces.get(self.norm(doc_num), [])

    # ----------------------------------------------------------------- query
    def norm(self, doc_num: str) -> str:
        return norm_doc_num(doc_num)

    def is_in_corpus(self, doc_num: str) -> bool:
        return self.norm(doc_num) in self.docs

    def lookup(self, doc_num: str) -> DocMeta | None:
        """Catalogue lookup (covers corpus + non-corpus laws)."""
        return self.catalog.get(self.norm(doc_num))

    def title_of(self, doc_num: str) -> str:
        m = self.catalog.get(self.norm(doc_num))
        return m.title if m else doc_num

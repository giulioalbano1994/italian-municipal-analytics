"""Variable catalog: reads resources/dizionario_variabili.csv and turns it into
(1) a compact block for the LLM prompt and (2) a synonym→column resolver.

Why: the LLM used to see only ~25 hardcoded synonyms, so it invented column names
that got dropped. Feeding it the REAL 82 columns + their synonyms makes almost every
variable answerable, and the resolver repairs near-misses.
"""
import csv
import re
import difflib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Rows flagged like this in the dictionary are technical/obsolete → skip.
_EXCLUDE = ("da escludere", "da rimuovere", "tecnico")

# Derived metrics computed in data_query._add_derived_metrics — not in the
# dictionary, so declare their synonyms here.
DERIVED = [
    {"var": "reddito_medio", "desc": "Reddito imponibile medio per contribuente (euro)",
     "syn": ["reddito medio", "reddito pro capite", "average income", "income per capita"]},
    {"var": "laureati_pct", "desc": "Laureati sul totale popolazione (%)",
     "syn": ["laureati percentuale", "quota laureati", "percentuale laureati", "graduates share"]},
    {"var": "imprese_attive_ratio", "desc": "Imprese attive su registrate (%)",
     "syn": ["imprese attive percentuale", "quota imprese attive", "tasso imprese attive"]},
    {"var": "saldo_migratorio_pct", "desc": "Saldo migratorio su popolazione (%)",
     "syn": ["saldo migratorio percentuale", "migrazione pro capite", "net migration rate"]},
]
_DERIVED_VARS = {d["var"] for d in DERIVED}

# Curated common Italian terms → real column (the dictionary's own synonyms miss
# some obvious ones, and population must resolve to pop_totale, not the obsolete
# 'popolazione'). Only applied when the target column exists.
EXTRA_SYNONYMS = {
    "popolazione": "pop_totale", "abitanti": "pop_totale", "residenti": "pop_totale", "pop": "pop_totale",
    "reddito": "reddito_imponibile_ammontare_in_euro", "redditi": "reddito_imponibile_ammontare_in_euro",
    "ricchezza": "reddito_imponibile_ammontare_in_euro", "contribuenti": "numero_contribuenti",
    "imposta": "imposta_netta_ammontare_in_euro",
    "laureati": "laureati_res_tot", "laureati donne": "laureati_res_femmine",
    "laureate": "laureati_res_femmine", "laureati uomini": "laureati_res_maschi",
    "gini": "gini_index", "disuguaglianza": "gini_index",
    "imprese": "imprese_attive_prov", "imprese registrate": "imprese_registrate_prov",
    "imprese attive": "imprese_attive_prov", "saldo migratorio": "saldo_migratorio_tot_com",
    "brevetti": "brevetti_num_prov", "merci": "merci_scaricate_tonnellate",
    "pensione": "reddito_da_pensione_ammontare_in_euro",
}


def load_catalog(path: str | Path) -> list[dict]:
    """Return [{var, desc, syn:[...]}] for usable variables, plus derived metrics."""
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                var = (r.get("variabile") or "").strip()
                if not var:
                    continue
                liv = (r.get("livello") or "").lower()
                desc = (r.get("descrizione") or "").strip()
                if any(h in liv or h in desc.lower() for h in _EXCLUDE):
                    continue
                raw_syn = r.get("sinonimi/termini collegati") or ""
                syn = [s.strip() for s in re.split(r"[;,]", raw_syn) if s.strip()]
                rows.append({"var": var, "desc": desc, "syn": syn})
    except Exception as e:
        logger.warning(f"catalog load failed ({path}): {e}")
    return rows + DERIVED


def prompt_lines(catalog: list[dict], columns) -> str:
    """Compact 'col: syn1, syn2' block for the prompt — only columns present in df."""
    cols = set(columns)
    out = []
    for r in catalog:
        if r["var"] in cols:
            syns = ", ".join(r["syn"][:4])
            out.append(f"- {r['var']}" + (f" ({syns})" if syns else ""))
    return "\n".join(out)


def synonym_index(catalog: list[dict], columns) -> dict:
    """Map every synonym and column name (lowercased) → real column."""
    cols = set(columns)
    idx = {}
    # Derived metrics exist after load even if absent from the raw CSV columns.
    for r in DERIVED:
        idx[r["var"].lower()] = r["var"]
        for s in r["syn"]:
            idx.setdefault(s.lower(), r["var"])
    for r in catalog:
        if r["var"] not in cols and r["var"] not in _DERIVED_VARS:
            continue
        idx[r["var"].lower()] = r["var"]
        for s in r["syn"]:
            idx.setdefault(s.lower(), r["var"])
    for term, col in EXTRA_SYNONYMS.items():
        if col in cols or col in _DERIVED_VARS:
            idx.setdefault(term, col)
    for c in cols:  # every real column resolves to itself
        idx.setdefault(c.lower(), c)
    return idx


def resolve(term: str, columns, syn_index: dict) -> str | None:
    """Best real column for a user/LLM term: exact → synonym → fuzzy (0.85)."""
    if not term:
        return None
    t = str(term).strip().lower()
    if t in syn_index:
        return syn_index[t]
    match = difflib.get_close_matches(t, list(syn_index.keys()), n=1, cutoff=0.85)
    return syn_index[match[0]] if match else None


if __name__ == "__main__":
    import pandas as pd
    cols = pd.read_csv("resources/df_ridotto_bot.csv", nrows=1).columns.str.lower().tolist()
    cat = load_catalog("resources/dizionario_variabili.csv")
    idx = synonym_index(cat, cols)
    print(f"catalog vars usable: {sum(1 for r in cat if r['var'] in set(cols))}/{len(cat)}")
    for term in ["reddito medio", "abitanti", "gini", "laureati donne", "quota laureati", "brevetti", "xyz"]:
        print(f"  {term!r:18} -> {resolve(term, cols, idx)}")

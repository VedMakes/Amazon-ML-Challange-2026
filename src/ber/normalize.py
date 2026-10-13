"""Deterministic, label-free representation bank.

Every function here is a fixed transform: no corpus statistics, no labels.
Raw fields are never overwritten; each view is a new column.

Views produced per record
  name_raw / addr_raw            V0 raw
  name_norm / addr_norm          V1 NFKD + accent strip + casefold + transliteration (V7) + punctuation cleanup
  name_tok / addr_tok            V2 token lists (legal-form and street-type aliases canonicalised)
  name_core                      V2' name tokens without legal-form / filler tokens (parallel view, never the only one)
  name_phon                      V7' phonetic tokens (metaphone, leading vowel stripped) for transliteration drift
  addr_num                       V4 numeric tokens, leading zeros stripped (raw zero-padded form kept in addr_tok)
  addr_alnum                     V5 intact alphanumeric tokens (c71, 12b, nh27)
  addr_postal                    V6 postal-like tokens (5-6 digit numbers)
  addr_house                     V6 first numeric token (weak house-number candidate)
  addr_nodesc                    V8 descriptor-stripped address (near/opp/behind/... + landmark removed)
"""
from __future__ import annotations

import re

import jellyfish
import polars as pl
from anyascii import anyascii

# ---------------------------------------------------------------------------
# Fixed, predeclared alias tables (chosen a priori, not learned from labels).
# Legal forms are a generic multinational list -- not restricted to US/India.
# ---------------------------------------------------------------------------
LEGAL_ALIAS = {
    "pvt": "private", "prv": "private", "pte": "private",
    "ltd": "limited", "ltda": "limited", "lmtd": "limited",
    "inc": "incorporated", "incorp": "incorporated",
    "corp": "corporation", "co": "company", "cos": "company", "cie": "company",
    "intl": "international", "mfg": "manufacturing", "mgmt": "management",
    "svcs": "services", "svc": "services", "bros": "brothers", "assoc": "associates",
    "grp": "group", "ent": "enterprises", "entp": "enterprises",
    # transliterated spellings of English legal forms (Devanagari -> Latin via anyascii)
    "praivet": "private", "praivt": "private", "limitid": "limited", "limitd": "limited",
    "elelpi": "llp", "elelsi": "llc", "kampani": "company", "korporeshan": "corporation",
    # French (predeclared; France is absent from training)
    "ets": "etablissements", "etabl": "etablissements", "soc": "societe", "ste": "societe", "sste": "societe",
}
LEGAL_FORMS = {
    "private", "limited", "incorporated", "corporation", "company", "llc", "llp", "lp", "plc",
    "pllc", "pc", "pa", "gmbh", "ag", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scop",
    "bv", "nv", "srl", "spa", "pty", "oy", "ab", "kg", "the", "and", "of", "dba", "aka",
    "sca", "scs", "selarl", "selas", "gie", "eirl", "sccv", "societe", "etablissements", "et", "de", "la", "le", "les", "du", "des",
}
ADDR_ALIAS = {
    # canonical forms are the abbreviations where a token is ambiguous across countries:
    # "st" = street (US/IN) = saint (FR); "ste" = suite (US) = sainte (FR) -- identical mapping on both sides
    "street": "st", "str": "st", "saint": "st", "suite": "ste", "sainte": "ste", "rd": "road", "ave": "avenue", "av": "avenue",
    "dr": "drive", "blvd": "boulevard", "bd": "boulevard", "ln": "lane", "ct": "court",
    "hwy": "highway", "pkwy": "parkway", "pl": "place", "sq": "square", "cir": "circle",
    "trl": "trail", "ter": "terrace", "tpke": "turnpike", "expy": "expressway",
    "apt": "apartment", "fl": "floor", "flr": "floor", "bldg": "building",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "opp": "opposite", "nr": "near", "mkt": "market", "ngr": "nagar", "clny": "colony",
    # French street types (La Poste style abbreviations), predeclared
    "bld": "boulevard", "boul": "boulevard", "che": "chemin", "chem": "chemin", "imp": "impasse",
    "rte": "route", "crs": "cours", "fbg": "faubourg", "fg": "faubourg", "res": "residence", "resid": "residence",
    "pte": "porte", "prom": "promenade", "qu": "quai", "rpt": "rondpoint", "lot": "lotissement",
    "r": "rue", "av.": "avenue", "all": "allee", "sq.": "square",
}
# words that are dropped from the core name but are not legal forms (never part of the legal-form key)
NAME_FILLERS = {"the", "and", "of", "dba", "aka", "et", "de", "la", "le", "les", "du", "des"}
DESCRIPTORS = ("near", "opposite", "opp", "behind", "beside", "besides", "next to", "adjacent to", "nr", "in front of")

_DESC_RE = r"\b(?:" + "|".join(re.escape(d) for d in DESCRIPTORS) + r")\b[^,]*"


def _translit_expr(col: pl.Expr) -> pl.Expr:
    """Transliterate any non-ASCII string with anyascii (accents -> base letter, Indic -> Latin).

    Applied to the raw string: stripping combining marks first would delete Indic vowel signs.
    """
    base = col.fill_null("")
    # Indic anusvara (Devanagari/Bengali/Gurmukhi/Gujarati/Oriya/Tamil/Telugu/Kannada/Malayalam)
    # is a nasal usually pronounced "n" before consonants; anyascii renders it "m".
    base = base.str.replace_all("[ंংਂੰંଂஂంಂം]", "n")
    return (
        pl.when(base.str.contains(r"[^\x00-\x7F]"))
        .then(base.map_elements(anyascii, return_dtype=pl.String))
        .otherwise(base)
    )


def _clean_expr(col: pl.Expr) -> pl.Expr:
    return (
        col.str.to_lowercase()
        .str.replace_all(r"https?://|www\.", " ")
        .str.replace_all(r"\.(com|net|org|in|co\.in|us|fr|biz|info|io)\b", " ")
        .str.replace_all(r"[&+]", " and ")
        .str.replace_all(r"\bd\s*/\s*b\s*/\s*a\b|\bdoing business as\b|\bt/a\b|\btrading as\b", " dba ")
        .str.replace_all(r"(\d)(st|nd|rd|th|eme|ere|er)\b", "$1")  # 5th / 15eme / 1er -> number (ordinal suffix)
        .str.replace_all(r"'", "")
        .str.replace_all(r"[^a-z0-9]+", " ")
        .str.strip_chars()
    )


def _alias_list(tok: pl.Expr, table: dict) -> pl.Expr:
    return tok.list.eval(pl.element().replace(table))


_PHON_CACHE: dict[str, str] = {}


def _phon(tok: str) -> str:
    p = _PHON_CACHE.get(tok)
    if p is None:
        p = jellyfish.metaphone(tok) if tok.isalpha() else tok
        p = p.lstrip("AEIOU") or p
        _PHON_CACHE[tok] = p
    return p


def phonetic_lists(tok_lists: pl.Series) -> pl.Series:
    """Metaphone per unique token (vectorised through a vocabulary map)."""
    flat = tok_lists.explode()
    uniq = flat.drop_nulls().unique()
    mapping = pl.DataFrame({"t": uniq, "p": [_phon(t) for t in uniq.to_list()]})
    df = pl.DataFrame({"row": pl.int_range(0, tok_lists.len(), eager=True), "t": tok_lists}).explode("t")
    df = df.join(mapping, on="t", how="left", maintain_order="left")
    out = df.group_by("row", maintain_order=True).agg(pl.col("p").drop_nulls())
    return out["p"]


def build_views(df: pl.DataFrame) -> pl.DataFrame:
    """df has entity_id, business_name, business_address, country. Returns df with all views added."""
    df = df.with_columns(
        _clean_expr(_translit_expr(pl.col("business_name"))).alias("name_norm"),
        _clean_expr(_translit_expr(pl.col("business_address"))).alias("addr_norm"),
        pl.col("business_name").fill_null("").str.contains(r"[^\x00-\x7F]").alias("name_nonascii"),
    )
    df = df.with_columns(
        _alias_list(pl.col("name_norm").str.split(" ").list.eval(pl.element().filter(pl.element() != "")), LEGAL_ALIAS).alias("name_tok"),
        _alias_list(pl.col("addr_norm").str.split(" ").list.eval(pl.element().filter(pl.element() != "")), ADDR_ALIAS).alias("addr_tok"),
    )
    df = df.with_columns(
        pl.col("name_tok").list.eval(pl.element().filter(~pl.element().is_in(list(LEGAL_FORMS)))).alias("name_core"),
        pl.col("addr_tok").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$")).str.strip_chars_start("0").replace("", "0")).alias("addr_num"),
        pl.col("addr_tok").list.eval(pl.element().filter(pl.element().str.contains(r"^(?:[a-z]+\d|\d+[a-z])[a-z0-9]*$"))).alias("addr_alnum"),
        pl.col("addr_tok").list.eval(pl.element().filter(pl.element().str.contains(r"^\d{5,6}$"))).alias("addr_postal"),
        _clean_expr(_translit_expr(pl.col("business_address")).str.to_lowercase().str.replace_all(_DESC_RE, " ")).alias("addr_nodesc"),
    )
    # alphanumerics like "c-71" are split by the cleaner; recover them from the raw address as an extra view
    df = df.with_columns(
        pl.concat_list(
            pl.col("addr_alnum"),
            _translit_expr(pl.col("business_address")).str.to_lowercase()
            .str.extract_all(r"\b[a-z]{1,3}\s*[-/]\s*#?\d+[a-z]?\b|\b\d+\s*[-/]\s*[a-z]\b")
            .list.eval(pl.element().str.replace_all(r"[^a-z0-9]", "")),
        ).list.unique(maintain_order=True).alias("addr_alnum"),
        pl.col("addr_num").list.first().alias("addr_house"),
        pl.col("name_core").list.sort().list.join(" ").alias("name_core_sorted"),
    )
    df = df.with_columns(phonetic_lists(df["name_core"]).alias("name_phon"))
    return df

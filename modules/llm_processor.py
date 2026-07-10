import os
import json
import logging
from enum import Enum
from dataclasses import dataclass
import google.generativeai as genai
from openai import OpenAI

from modules import catalog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# ENUM E DATACLASS
# ---------------------------------------------------------------------

class QueryType(Enum):
    SINGLE_COMUNE = "single_comune"
    TIME_SERIES = "time_series"
    CROSS_SECTION = "cross_section"
    COMPARISON = "comparison"
    RANKING = "ranking"          # top/bottom N by a metric


class ChartType(Enum):
    BAR = "bar"
    BARH = "barh"                # horizontal bars — readable rankings
    LINE = "line"
    PIE = "pie"
    MAP = "map"


@dataclass
class QueryParameters:
    comuni: list[str] | None = None
    metrics: list[str] | None = None
    query_type: QueryType | None = None
    chart_type: ChartType | None = None
    start_year: int | None = None
    end_year: int | None = None
    anno: int | None = None
    top_n: int | None = None                 # ranking size
    ascending: bool = False                  # ranking direction (False = top/highest)
    level: str | None = None                 # aggregation: comune | provincia | regione


# ---------------------------------------------------------------------
# LLM PROCESSOR
# ---------------------------------------------------------------------

class LLMProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key.strip() if api_key else None
        self.use_openai = bool(api_key and api_key.startswith("sk-"))
        self.model_name = "gpt-4o-mini" if self.use_openai else "gemini-1.5-flash"

        if self.use_openai:
            logger.info(f"🤖 OpenAI attivo (modello {self.model_name})")
            self.client = OpenAI(api_key=self.api_key)
        else:
            logger.info(f"🌐 Gemini attivo (modello {self.model_name})")
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)

        # Legacy synonym map (kept as a base; the real catalog is loaded below)
        self.metric_mapping = self._build_metric_mapping()

        # Variable catalog (dictionary) → injected into the prompt + resolver.
        dict_path = os.getenv("VARIABLES_DICT", "resources/dizionario_variabili.csv")
        self.catalog = catalog.load_catalog(dict_path)
        self.columns = []          # real df columns (set via set_context after load)
        self.comuni_list = []
        self.syn_index = {}
        self._catalog_block = ""
        self._parse_cache = {}     # normalized question -> QueryParameters

    # -----------------------------------------------------------------
    def set_context(self, columns, comuni):
        """Called once after the dataframe is loaded: give the LLM the real column
        names (for the prompt) and the resolver its synonym index. Avoids re-reading
        the 90 MB CSV just to list comuni/columns."""
        self.columns = list(columns or [])
        self.comuni_list = list(comuni or [])
        self.syn_index = catalog.synonym_index(self.catalog, self.columns)
        self._catalog_block = catalog.prompt_lines(self.catalog, self.columns)
        self._parse_cache.clear()
        logger.info(f"🧭 LLM context: {len(self.columns)} colonne, {len(self.comuni_list)} comuni, "
                    f"{len(self.syn_index)} sinonimi indicizzati")

    def resolve_metric(self, term: str):
        """Map an LLM/user term to a real column, or None."""
        return catalog.resolve(term, self.columns, self.syn_index)

    # -----------------------------------------------------------------
    def _build_metric_mapping(self):
        """
        Mappa semantica aggiornata per dataset socioeconomico comunale.
        """
        return {
            # Reddito e fisco
            "reddito": "reddito_imponibile_ammontare_in_euro",
            "redditi": "reddito_imponibile_ammontare_in_euro",
            "ricchezza": "reddito_imponibile_ammontare_in_euro",
            "contribuenti": "numero_contribuenti",
            "imposta": "imposta_netta_ammontare_in_euro",
            "addizionale": "addizionale_regionale_dovuta_ammontare_in_euro",

            # Popolazione
            "popolazione": "popolazione",
            "abitanti": "popolazione",
            "residenti": "popolazione",

            # Istruzione
            "laureati": "laureati_res_tot",
            "laureati donne": "laureati_res_femmine",
            "laureati uomini": "laureati_res_maschi",

            # Disuguaglianza
            "gini": "gini_index",
            "disuguaglianza": "gini_index",

            # Imprese
            "imprese registrate": "imprese_registrate_prov",
            "imprese attive": "imprese_attive_prov",
            "imprese saldo": "imprese_saldo_prov",

            # Migrazioni
            "saldo migratorio": "saldo_migratorio_tot_com",
            "saldo estero": "saldo_migratorio_estero_com",

            # Innovazione
            "brevetti": "brevetti_num_prov",
            "brevetti percentuale": "brevetti_pct_prov",

            # Merci e traffico
            "merci": "merci_scaricate_tonnellate",
        }

    # -----------------------------------------------------------------
    def _prompt_template(self):
        """Prompt with the REAL variable catalog so the model returns valid columns."""
        vars_block = self._catalog_block or "(catalogo non disponibile)"
        return (
            "Sei un assistente che traduce richieste in parametri strutturati per un motore di grafici "
            "su dati socio-economici dei comuni italiani.\n"
            'Rispondi SOLO con JSON valido (virgolette doppie), formato:\n'
            '{"comuni": [], "metrics": [], "query_type": "", "chart_type": "", '
            '"start_year": null, "end_year": null, "anno": null, '
            '"top_n": null, "ascending": false, "level": null}\n\n'
            "IMPORTANTE: in 'metrics' usa ESCLUSIVAMENTE i nomi colonna elencati qui sotto "
            "(scegli quelli più pertinenti alla domanda). Non inventare nomi.\n\n"
            "COLONNE DISPONIBILI (nome: sinonimi):\n"
            f"{vars_block}\n\n"
            "Regole:\n"
            "- 'nel tempo/andamento/evoluzione' → query_type 'time_series', chart_type 'line'.\n"
            "- confronto tra più città → 'comparison', 'bar'.\n"
            "- un solo comune senza anni → 'cross_section'.\n"
            "- 'classifica/top/i primi/i più .../i migliori' → query_type 'ranking', chart_type 'barh', "
            "top_n = numero richiesto (default 10), ascending=false; per 'i meno/i peggiori/i più bassi' ascending=true.\n"
            "- aggregazione territoriale: 'per regione'/'tra le regioni' → level 'regione'; "
            "'per provincia' → level 'provincia'; altrimenti level 'comune'.\n"
            "- 'mappa' o 'distribuzione territoriale' → 'map'.\n"
            "- percentuale/quota/pro capite → usa le colonne derivate (reddito_medio, laureati_pct, "
            "imprese_attive_ratio, saldo_migratorio_pct) se pertinenti.\n"
            "- anni = numeri interi; 'comuni' con l'iniziale maiuscola.\n"
        )

    # -----------------------------------------------------------------
    def process_request(self, text: str) -> QueryParameters:
        """
        Elabora una richiesta utente (testo) e restituisce i parametri strutturati.
        """
        logger.info(f"🧠 Elaborazione query utente: {text}")

        cache_key = " ".join((text or "").lower().split())
        if cache_key in self._parse_cache:
            logger.info("⚡ Parse da cache")
            return self._parse_cache[cache_key]

        prompt = self._prompt_template()

        if self.use_openai:
            logger.info(f"🚀 Invio a OpenAI ({self.model_name})")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
                max_tokens=300,
            )
            content = response.choices[0].message.content
        else:
            logger.info(f"🚀 Invio a Gemini ({self.model_name})")
            response = self.model.generate_content(prompt + "\n\nUtente: " + text)
            content = response.text

        try:
            content_clean = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(content_clean)
            logger.info(f"🧾 Risposta grezza dal modello: {content_clean[:200]}...")
        except Exception as e:
            logger.warning(f"⚠️ Parsing JSON fallito: {e} | Testo: {content}")
            parsed = {}

        # Map every returned metric to a REAL column (repairs near-misses); drop
        # what can't be resolved so a bad name never silently empties the result.
        raw_metrics = parsed.get("metrics") or []
        resolved, unknown = [], []
        for m in raw_metrics:
            col = self.resolve_metric(m)
            (resolved if col else unknown).append(col or m)
        # dedup, preserve order
        resolved = list(dict.fromkeys(resolved))
        if unknown:
            logger.info(f"⚠️ Metriche non risolte: {unknown}")

        def _to_int(v):
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        def _enum(E, val, default):
            try:
                return E(val)
            except (ValueError, TypeError):
                return default

        level = parsed.get("level")
        level = level.lower() if isinstance(level, str) and level.lower() in (
            "comune", "provincia", "regione") else None

        params = QueryParameters(
            comuni=parsed.get("comuni", []),
            metrics=resolved,
            query_type=_enum(QueryType, parsed.get("query_type"), QueryType.TIME_SERIES),
            chart_type=_enum(ChartType, parsed.get("chart_type"), ChartType.LINE),
            start_year=_to_int(parsed.get("start_year")),
            end_year=_to_int(parsed.get("end_year")),
            anno=_to_int(parsed.get("anno")),
            top_n=_to_int(parsed.get("top_n")),
            ascending=bool(parsed.get("ascending", False)),
            level=level,
        )

        logger.info(f"✅ Parametri estratti: {params}")
        self._parse_cache[cache_key] = params
        return params

    # -----------------------------------------------------------------
    def generate_commentary(self, df, params: QueryParameters) -> str:
        """
        Crea un breve commento descrittivo sui dati (trend, confronti, variazioni percentuali).
        """
        try:
            if df is None or df.empty:
                return ""
            num = df.select_dtypes("number").drop(columns=["anno"], errors="ignore")
            if num.empty:
                return ""
            bits = []
            for c in num.columns[:4]:
                s = num[c].dropna()
                if s.empty:
                    continue
                last, first = s.iloc[-1], s.iloc[0]
                arrow = "📈" if last > first else "📉" if last < first else "➡️"
                bits.append(f"*{c}*: {last:,.0f} {arrow}".replace(",", "."))
            return "🔎 " + "  |  ".join(bits) if bits else ""
        except Exception:
            return ""

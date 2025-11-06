import os
import re
import json
import logging
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
from openai import OpenAI

logger = logging.getLogger(__name__)

# -----------------------------
# ENUMS
# -----------------------------
class QueryType(Enum):
    SINGLE_YEAR = "single_year"
    TIME_SERIES = "time_series"
    CROSS_SECTION = "cross_section"
    SINGLE_COMUNE = "single_comune"

class ChartType(Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    MAP = "map"

# -----------------------------
# DATA CLASS
# -----------------------------
@dataclass
class QueryParameters:
    comuni: List[str] = None
    metrics: List[str] = None
    query_type: QueryType = QueryType.CROSS_SECTION
    chart_type: ChartType = ChartType.BAR
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    anno: Optional[int] = None

# -----------------------------
# LLM PROCESSOR
# -----------------------------
class LLMProcessor:
    """
    Interprete ibrido:
    1️⃣ tenta il parsing locale rule-based (comuni, metriche, periodo)
    2️⃣ se confidenza < 0.8, chiede a OpenAI di generare i parametri JSON
    3️⃣ restituisce sempre QueryParameters coerente
    """

    def __init__(self, api_key: Optional[str] = None, df_path: str = "resources/df_ridotto_bot.csv"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.metric_mapping = self._build_metric_mapping()
        self.comuni_list = self._load_comuni(df_path)

    # -----------------------------
    # Public
    # -----------------------------
    def process_request(self, user_request: str) -> QueryParameters:
        """Parsing locale + fallback GPT"""
        text = (user_request or "").strip()
        local_params, confidence = self._local_parse(text)

        if confidence >= 0.8 or not self.client:
            logger.info(f"✅ Parsing locale (confidence={confidence:.2f})")
            return local_params

        # fallback LLM
        logger.info(f"🤖 Invio a LLM (confidence={confidence:.2f})")
        prompt = self._build_llm_prompt(text)
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": text}],
                temperature=0.2,
            )
            content = response.choices[0].message.content
            params = self._parse_llm_response(content)
            if params:
                logger.info("✅ Risposta LLM interpretata correttamente")
                return params
        except Exception as e:
            logger.warning(f"⚠️ Errore nel parsing LLM: {e}")

        logger.info("🔁 Ritorno al fallback locale")
        return local_params

    # -----------------------------
    # Local parsing (rule-based)
    # -----------------------------
    def _local_parse(self, text: str) -> Tuple[QueryParameters, float]:
        """Parsing semplice ma robusto basato su regole"""
        t = text.lower()
        params = QueryParameters(comuni=[], metrics=[])

        # --- riconosci comuni ---
        comuni_trovati = [c for c in self.comuni_list if c.lower() in t]
        if comuni_trovati:
            params.comuni = comuni_trovati

        # --- metriche ---
        found_metric = None
        for k, v in self.metric_mapping.items():
            if k in t:
                found_metric = v
                params.metrics = [v]
                break

        # --- tipo di query ---
        if "nel tempo" in t or "serie" in t or "andamento" in t:
            params.query_type = QueryType.TIME_SERIES
            params.chart_type = ChartType.LINE
        elif "mappa" in t or "region" in t or "provinc" in t:
            params.query_type = QueryType.CROSS_SECTION
            params.chart_type = ChartType.MAP
        else:
            params.query_type = QueryType.CROSS_SECTION
            params.chart_type = ChartType.BAR

        # --- periodo ---
        anni = re.findall(r"20\d{2}", t)
        if len(anni) == 1:
            params.anno = int(anni[0])
        elif len(anni) >= 2:
            params.start_year, params.end_year = int(anni[0]), int(anni[1])

        # --- confidence ---
        conf = 0.0
        if found_metric:
            conf += 0.4
        if comuni_trovati:
            conf += 0.4
        if "tempo" in t or "anni" in t:
            conf += 0.2

        return params, min(conf, 1.0)

    # -----------------------------
    # Prompt builder for GPT
    # -----------------------------
    def _build_llm_prompt(self, text: str) -> str:
        """
        Costruisce un prompt chiaro per l’LLM.
        Il modello deve restituire solo un JSON strutturato.
        """
        examples = """
ESEMPI DI RICHIESTE E RISPOSTE CORRETTE:

Utente: "Popolazione Bari e Napoli nel tempo"
Risposta:
{"comuni":["Bari","Napoli"],"metrics":["pop_totale"],"query_type":"time_series","chart_type":"line"}

Utente: "Reddito medio Torino 2015-2023"
Risposta:
{"comuni":["Torino"],"metrics":["average_income"],"query_type":"time_series","chart_type":"line","start_year":2015,"end_year":2023}

Utente: "Gini index Firenze ultimo anno"
Risposta:
{"comuni":["Firenze"],"metrics":["gini_index"],"query_type":"cross_section","chart_type":"bar"}

Utente: "Quota pensionati Roma e Milano"
Risposta:
{"comuni":["Roma","Milano"],"metrics":["pensionati_percentuale"],"query_type":"cross_section","chart_type":"bar"}

Utente: "Laureati residenti Bologna"
Risposta:
{"comuni":["Bologna"],"metrics":["laureati_percentuale"],"query_type":"cross_section","chart_type":"bar"}

Utente: "Imprese attive Napoli nel tempo"
Risposta:
{"comuni":["Napoli"],"metrics":["imprese_attive"],"query_type":"time_series","chart_type":"line"}

Utente: "Confronto redditi tra Milano, Roma e Torino nel 2022"
Risposta:
{"comuni":["Milano","Roma","Torino"],"metrics":["average_income"],"query_type":"single_year","chart_type":"bar","anno":2022}
"""
        available_metrics = ", ".join(sorted(set(self.metric_mapping.values())))
        return (
            "Sei un parser di richieste in linguaggio naturale per dati socio-economici comunali italiani.\n"
            "Il tuo compito è estrarre le seguenti informazioni e restituirle SOLO come JSON valido:\n"
            " - comuni (lista di nomi di città)\n"
            " - metrics (lista di variabili disponibili)\n"
            " - query_type (time_series, cross_section, single_year)\n"
            " - chart_type (line, bar, map)\n"
            " - start_year, end_year o anno se presenti\n\n"
            f"Variabili disponibili: {available_metrics}\n\n"
            f"{examples}\n\n"
            "Rispondi SOLO con JSON. Nessun commento, nessun testo fuori dal JSON."
        )

    # -----------------------------
    # Parse LLM response
    # -----------------------------
    def _parse_llm_response(self, content: str) -> Optional[QueryParameters]:
        try:
            json_str = re.search(r"\{.*\}", content, re.DOTALL)
            if not json_str:
                return None
            data = json.loads(json_str.group(0))
            params = QueryParameters()
            params.comuni = data.get("comuni", [])
            params.metrics = data.get("metrics", [])
            params.query_type = QueryType(data.get("query_type", "cross_section"))
            params.chart_type = ChartType(data.get("chart_type", "bar"))
            params.start_year = data.get("start_year")
            params.end_year = data.get("end_year")
            params.anno = data.get("anno")
            return params
        except Exception as e:
            logger.warning(f"Errore parsing JSON LLM: {e}")
            return None

    # -----------------------------
    # Helpers
    # -----------------------------
    def _load_comuni(self, df_path: str) -> List[str]:
        try:
            df = pd.read_csv(df_path, usecols=["comune"])
            comuni = sorted(df["comune"].dropna().unique().tolist())
            logger.info(f"🗺️ Comuni caricati: {len(comuni)}")
            return comuni
        except Exception as e:
            logger.warning(f"⚠️ Errore caricamento comuni: {e}")
            return []

    def _build_metric_mapping(self) -> dict:
        return {
            "popolazione": "pop_totale",
            "abitanti": "pop_totale",
            "reddito": "average_income",
            "redditi": "average_income",
            "ricchezza": "average_income",
            "gini": "gini_index",
            "disuguaglianza": "gini_index",
            "pensionati": "pensionati_percentuale",
            "pensione": "pensionati_percentuale",
            "laureati": "laureati_percentuale",
            "imprese": "imprese_attive",
            "aziende": "imprese_attive",
            "tasse": "tax_revenue_per_capita",
            "entrate": "tax_revenue_per_capita"
        }


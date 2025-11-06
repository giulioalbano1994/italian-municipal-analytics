import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from modules import QueryParameters, QueryType, ChartType
import logging

logger = logging.getLogger(__name__)

# ==========================================================
#  DataFrameManager - versione ibrida robusta
# ==========================================================
class DataFrameManager:
    def __init__(self, data_dir: str = "resources"):
        self.data_dir = Path(data_dir)
        self.df = None
        self._meta = {}

    # ------------------------------------------------------
    # Caricamento dati
    # ------------------------------------------------------
    def load_data(self):
        start = time.time()
        csv_paths = list(self.data_dir.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"Nessun file CSV trovato in {self.data_dir}")

        p = csv_paths[0]
        df = pd.read_csv(p, low_memory=False)
        self.df = self._normalize_df(df)
        elapsed = round(time.time() - start, 2)
        logger.info(f"✅ Dati caricati da: {p} | righe={len(df)} | colonne={len(df.columns)} | tempo={elapsed}s")
        self._meta = self._build_meta(self.df)
        return self.df

    # ------------------------------------------------------
    # Normalizzazione nomi colonne
    # ------------------------------------------------------
    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip().str.lower()
        rename_map = {
            "anno_rif": "anno",
            "nome_comune": "comune",
            "cod_comune": "codice_comune"
        }
        df = df.rename(columns=rename_map)
        return df

    # ------------------------------------------------------
    # Informazioni meta (fonti, copertura, anno)
    # ------------------------------------------------------
    def _build_meta(self, df: pd.DataFrame) -> dict:
        latest_year = int(df["anno"].max()) if "anno" in df else None
        sources = [c for c in df.columns if any(x in c for x in ["istat", "mef", "miur", "infocamere", "eurostat"])]
        coverage_str = f"{len(df):,}x{len(df.columns)}".replace(",", ".")
        return {"latest_year": latest_year, "sources": sources, "coverage_str": coverage_str}

    def dataset_meta(self) -> dict:
        return self._meta

    # ------------------------------------------------------
    # Query principale: restituisce (df, xlabel, ylabel, meta)
    # ------------------------------------------------------
    def query_data(self, params: QueryParameters):
        if self.df is None:
            raise ValueError("Dataset non caricato. Esegui load_data() prima di query_data().")

        df = self.df.copy()
        df = self._filter_comuni(df, params)
        df = self._filter_period(df, params)
        df = self._filter_metrics(df, params)

        if df.empty:
            logger.warning("⚠️ Nessun dato trovato dopo i filtri applicati.")
            return df, "", "", self._meta

        xlabel, ylabel = self._infer_labels(params)
        meta = self._meta
        return df, xlabel, ylabel, meta

    # ------------------------------------------------------
    # Filtri principali
    # ------------------------------------------------------
    def _filter_comuni(self, df: pd.DataFrame, params: QueryParameters) -> pd.DataFrame:
        if "comune" not in df.columns:
            return df
        if not params.comuni:
            return df
        comuni_lower = [c.lower() for c in params.comuni]
        mask = df["comune"].str.lower().isin(comuni_lower)
        return df[mask]

    def _filter_period(self, df: pd.DataFrame, params: QueryParameters) -> pd.DataFrame:
        if "anno" not in df.columns:
            return df
        if params.anno:
            return df[df["anno"] == params.anno]
        if params.start_year and params.end_year:
            return df[(df["anno"] >= params.start_year) & (df["anno"] <= params.end_year)]
        # fallback: prendi ultimi 10 anni se disponibili
        if df["anno"].nunique() > 10:
            maxy = int(df["anno"].max())
            return df[df["anno"] >= maxy - 10]
        return df

    def _filter_metrics(self, df: pd.DataFrame, params: QueryParameters) -> pd.DataFrame:
        metrics = params.metrics or []
        if not metrics:
            # fallback intelligente: cerca colonne tipiche
            if any(w in params.query_text.lower() for w in ["popolazione", "abitanti"]):
                metrics = ["pop_totale"]
            elif any(w in params.query_text.lower() for w in ["reddito", "income"]):
                metrics = ["average_income"]
            else:
                # ultima risorsa: prime 3 colonne numeriche
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                metrics = numeric_cols[:3]
        df = df.loc[:, [c for c in df.columns if c in metrics or c in ["comune", "anno"]]]
        params.metrics = metrics
        return df

    # ------------------------------------------------------
    # Etichette e descrizioni automatiche
    # ------------------------------------------------------
    def _infer_labels(self, params: QueryParameters):
        if params.chart_type == ChartType.BAR:
            xlabel = "Comuni"
            ylabel = ", ".join(params.metrics or ["Valore"])
        elif params.chart_type == ChartType.LINE:
            xlabel = "Anno"
            ylabel = ", ".join(params.metrics or ["Valore"])
        else:
            xlabel = "Categoria"
            ylabel = "Valore"
        return xlabel, ylabel

    # ------------------------------------------------------
    # Query per mappe (regioni / comuni)
    # ------------------------------------------------------
    def query_data_for_map(self, params: QueryParameters):
        if self.df is None:
            raise ValueError("Dataset non caricato.")
        df = self.df.copy()
        metric = params.metrics[0] if params.metrics else None
        if not metric or metric not in df.columns:
            raise ValueError(f"Metrica non trovata: {metric}")

        if "regione" in df.columns:
            group = "regione"
        elif "provincia" in df.columns:
            group = "provincia"
        else:
            group = "comune"

        anno = params.anno or df["anno"].max()
        df = df[df["anno"] == anno]
        df_map = df.groupby(group, as_index=False)[metric].mean()
        xlabel = group.capitalize()
        ylabel = metric
        meta = self._meta | {"map_level": group}
        return df_map, xlabel, ylabel, meta

    # ------------------------------------------------------
    # Variabili disponibili
    # ------------------------------------------------------
    def available_variables(self, limit: int = 200):
        if self.df is None:
            return []
        cols = self.df.columns.tolist()
        return cols[:limit]

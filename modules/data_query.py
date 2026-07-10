import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from modules import QueryParameters, QueryType, ChartType
import logging

logger = logging.getLogger(__name__)

# ==========================================================
#  DataFrameManager - versione completa e coerente (v3)
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
        df = self._normalize_df(df)
        df = self._add_derived_metrics(df)

        self.df = df
        elapsed = round(time.time() - start, 2)
        logger.info(f"✅ Dati caricati da: {p} | righe={len(df)} | colonne={len(df.columns)} | tempo={elapsed}s")
        self._meta = self._build_meta(df)
        return df

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
        # uniforma tipo anno
        if "anno" in df.columns:
            df["anno"] = pd.to_numeric(df["anno"], errors="coerce").astype("Int64")
        # chiave comune minuscola precalcolata: evita str.lower() su 195k righe a ogni query
        if "comune" in df.columns:
            df["comune_norm"] = df["comune"].astype(str).str.lower()
        return df

    # ------------------------------------------------------
    # Aggiunge metriche derivate (pro capite, percentuali, rapporti)
    # ------------------------------------------------------
    def _add_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Reddito medio
        if all(c in df.columns for c in ["reddito_imponibile_ammontare_in_euro", "numero_contribuenti"]):
            df["reddito_medio"] = (
                df["reddito_imponibile_ammontare_in_euro"] / df["numero_contribuenti"]
            ).replace([np.inf, -np.inf], np.nan)

        # Laureati su popolazione (%)
        if all(c in df.columns for c in ["laureati_res_tot", "popolazione"]):
            df["laureati_pct"] = (
                df["laureati_res_tot"] / df["popolazione"] * 100
            ).replace([np.inf, -np.inf], np.nan)

        # Imprese attive su registrate (%)
        if all(c in df.columns for c in ["imprese_attive_prov", "imprese_registrate_prov"]):
            df["imprese_attive_ratio"] = (
                df["imprese_attive_prov"] / df["imprese_registrate_prov"] * 100
            ).replace([np.inf, -np.inf], np.nan)

        # Saldo migratorio su popolazione (%)
        if all(c in df.columns for c in ["saldo_migratorio_tot_com", "popolazione"]):
            df["saldo_migratorio_pct"] = (
                df["saldo_migratorio_tot_com"] / df["popolazione"] * 100
            ).replace([np.inf, -np.inf], np.nan)

        return df

    # ------------------------------------------------------
    # Informazioni meta (fonti, copertura, anno)
    # ------------------------------------------------------
    def _build_meta(self, df: pd.DataFrame) -> dict:
        latest_year = int(df["anno"].dropna().max()) if "anno" in df.columns else None
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

        # No full .copy(): boolean filters already return new frames, so self.df
        # is never mutated. Copying 200 MB per request was pure waste.
        df = self.df
        df = self._filter_comuni(df, params)
        df = self._filter_period(df, params)
        df = self._select_metrics(df, params)  # long: [comune, anno, *metrics]

        if df.empty:
            logger.warning("⚠️ Nessun dato trovato dopo i filtri applicati.")
            return df, "", "", self._meta

        wide, x_col = self._shape_for_chart(df, params)
        xlabel, ylabel = self._labels(x_col, df, params)
        return wide, xlabel, ylabel, self._meta

    # ------------------------------------------------------
    # Reshape long → wide, chart-ready. NEVER leaves the string 'comune'
    # column as a data series (that was the phantom "= 0" line).
    # ------------------------------------------------------
    def _shape_for_chart(self, df: pd.DataFrame, params: QueryParameters):
        metrics = [c for c in df.columns if c not in ("comune", "anno")]
        if not metrics:
            return df, df.columns[0]
        m0 = metrics[0]
        n_comuni = df["comune"].nunique() if "comune" in df.columns else 1
        n_years = df["anno"].nunique() if "anno" in df.columns else 1

        if n_comuni > 1:
            if n_years > 1:  # time series: one column (line) per comune
                wide = (df.pivot_table(index="anno", columns="comune", values=m0, aggfunc="mean")
                          .sort_index().reset_index())
                return wide, "anno"
            # single year: one bar per comune
            return df.groupby("comune", as_index=False)[m0].mean(), "comune"

        # single comune (or aggregate)
        if n_years > 1:  # time series of the metric(s), no comune column
            wide = df.set_index("anno")[metrics].sort_index().reset_index()
            return wide, "anno"
        # single comune + single year → compare the metrics as bars
        vals = df[metrics].mean()
        return pd.DataFrame({"metrica": vals.index, "valore": vals.values}), "metrica"

    def _labels(self, x_col: str, df: pd.DataFrame, params: QueryParameters):
        xlabel = {"anno": "Anno", "comune": "Comune", "metrica": "Metrica"}.get(x_col, x_col.capitalize())
        ylabel = ", ".join(params.metrics or []) or "Valore"
        return xlabel, ylabel

    # ------------------------------------------------------
    # Filtri principali
    # ------------------------------------------------------
    def _filter_comuni(self, df: pd.DataFrame, params: QueryParameters) -> pd.DataFrame:
        if "comune" not in df.columns or not params.comuni:
            return df
        comuni_lower = [c.lower() for c in params.comuni]
        key = df["comune_norm"] if "comune_norm" in df.columns else df["comune"].str.lower()
        return df[key.isin(comuni_lower)]

    def _filter_period(self, df: pd.DataFrame, params: QueryParameters) -> pd.DataFrame:
        if "anno" not in df.columns:
            return df

        if params.anno:
            df = df[df["anno"] == params.anno]
        elif params.start_year and params.end_year:
            df = df[(df["anno"] >= params.start_year) & (df["anno"] <= params.end_year)]
        else:
            # fallback: ultimi 10 anni se disponibili
            maxy = int(df["anno"].dropna().max())
            if df["anno"].nunique() > 10:
                df = df[df["anno"] >= maxy - 10]
        return df

    def _select_metrics(self, df: pd.DataFrame, params: QueryParameters) -> pd.DataFrame:
        metrics = params.metrics or []
        keep_cols = ["comune", "anno"]

        # fallback intelligente
        if not metrics:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            metrics = numeric_cols[:1]  # prendi la prima numerica utile
            params.metrics = metrics

        valid_metrics = [m for m in metrics if m in df.columns]
        keep_cols.extend(valid_metrics)
        # small result → detach with copy so downstream never touches self.df
        return df[keep_cols].copy()

    # ------------------------------------------------------
    # Etichette automatiche
    # ------------------------------------------------------
    def _infer_labels(self, params: QueryParameters):
        if params.chart_type == ChartType.BAR:
            xlabel = "Comune"
            ylabel = ", ".join(params.metrics or ["Valore"])
        elif params.chart_type == ChartType.LINE:
            xlabel = "Anno"
            ylabel = ", ".join(params.metrics or ["Valore"])
        else:
            xlabel = "Categoria"
            ylabel = "Valore"
        return xlabel, ylabel

    # ------------------------------------------------------
    # Query per mappe
    # ------------------------------------------------------
    def query_data_for_map(self, params: QueryParameters):
        if self.df is None:
            raise ValueError("Dataset non caricato.")
        df = self.df
        metric = params.metrics[0] if params.metrics else None
        if not metric or metric not in df.columns:
            raise ValueError(f"Metrica non trovata: {metric}")

        group = (
            "regione"
            if "regione" in df.columns
            else "provincia"
            if "provincia" in df.columns
            else "comune"
        )

        anno = params.anno or df["anno"].max()
        df = df[df["anno"] == anno]
        df_map = df.groupby(group, as_index=False)[metric].mean()

        xlabel = group.capitalize()
        ylabel = metric
        meta = self._meta | {"map_level": group}
        return df_map, xlabel, ylabel, meta

    # ------------------------------------------------------
    # Classifiche (top/bottom N), con aggregazione territoriale
    # ------------------------------------------------------
    def _level_column(self, level: str) -> str:
        candidates = {
            "regione": ["regione_mef", "regione"],
            "provincia": ["provincia_label", "provincia", "sigla_provincia"],
            "comune": ["comune"],
        }.get(level or "comune", ["comune"])
        return next((c for c in candidates if c in self.df.columns), "comune")

    def query_ranking(self, params: QueryParameters):
        if self.df is None:
            raise ValueError("Dataset non caricato.")
        metric = params.metrics[0] if params.metrics else None
        if not metric or metric not in self.df.columns:
            return pd.DataFrame(), "", "", self._meta

        level = params.level or "comune"
        col = self._level_column(level)
        year = params.anno or int(self.df["anno"].dropna().max())

        df = self.df[self.df["anno"] == year]
        agg = (df.groupby(col, as_index=False)[metric].mean()
                 .dropna(subset=[metric])
                 .sort_values(metric, ascending=bool(params.ascending))
                 .head(params.top_n or 10))
        # rename to a name the chart recognises as the category axis
        agg = agg.rename(columns={col: level})[[level, metric]]
        meta = self._meta | {"rank_year": year, "level": level}
        return agg, level.capitalize(), metric, meta

    # ------------------------------------------------------
    # Variabili disponibili
    # ------------------------------------------------------
    def available_variables(self, limit: int = 200):
        if self.df is None:
            return []
        return self.df.columns[:limit].tolist()

    def comuni_list(self):
        """Sorted unique comune names, from the already-loaded df (no CSV re-read)."""
        if self.df is None or "comune" not in self.df.columns:
            return []
        return sorted(self.df["comune"].dropna().astype(str).unique().tolist())

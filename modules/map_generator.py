import io
import os
import json
import math
import logging
import unicodedata

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

logger = logging.getLogger(__name__)

# Regional choropleth WITHOUT geopandas: parse a GeoJSON of Italian regions and
# draw the polygons with matplotlib. Join is by normalized region name.
DEFAULT_GEOJSON = os.path.join("resources", "geo", "regioni.geojson")
INK = "#1f2937"
MUTE = "#6b7280"


def _norm(s: str) -> str:
    """'VALLE D'AOSTA' and 'Valle d'Aosta/Vallée d'Aoste' → 'valle d aosta'-ish match key."""
    s = str(s).split("/")[0].strip().lower().replace("-", " ")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return " ".join(s.split())


def _exterior_rings(geom: dict):
    t = geom.get("type")
    c = geom.get("coordinates", [])
    if t == "Polygon":
        return [c[0]] if c else []
    if t == "MultiPolygon":
        return [poly[0] for poly in c if poly]
    return []


class MapGenerator:
    def __init__(self):
        self.geojson_path = os.getenv("IT_REGIONI_GEOJSON", DEFAULT_GEOJSON)
        self._features = None

    def _load(self):
        if self._features is None:
            with open(self.geojson_path, encoding="utf-8") as f:
                self._features = json.load(f)["features"]
        return self._features

    def available(self) -> bool:
        return os.path.exists(self.geojson_path)

    def generate_choropleth(self, df, metric_col: str, level: str = "regione",
                            title: str = "", subtitle: str = "") -> bytes:
        """df: [<region col>, metric]. Renders an Italy regional choropleth."""
        if not self.available():
            return self._fallback_bar(df, metric_col, title, subtitle)

        features = self._load()
        name_col = df.columns[0]
        values = {_norm(r[name_col]): float(r[metric_col])
                  for _, r in df.dropna(subset=[metric_col]).iterrows()}

        patches, vals = [], []
        for feat in features:
            key = _norm(feat["properties"].get("reg_name", ""))
            v = values.get(key, np.nan)
            for ring in _exterior_rings(feat["geometry"]):
                patches.append(PathPatch(Path(np.asarray(ring))))
                vals.append(v)

        vals = np.array(vals, dtype=float)
        finite = vals[np.isfinite(vals)]
        norm = Normalize(vmin=finite.min(), vmax=finite.max()) if finite.size else Normalize(0, 1)
        cmap = cm.get_cmap("viridis").copy()
        cmap.set_bad("#e5e7eb")  # regions without data → light grey

        fig, ax = plt.subplots(figsize=(7.2, 8.4), dpi=200)
        pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="white", linewidth=0.5)
        pc.set_array(vals)
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_aspect(1.32)  # Italy ~42°N: makes the boot look natural
        ax.axis("off")

        cbar = fig.colorbar(pc, ax=ax, fraction=0.035, pad=0.02, shrink=0.7)
        cbar.ax.tick_params(labelsize=8, colors=MUTE)
        cbar.outline.set_visible(False)

        fig.suptitle(title, fontsize=15, fontweight="bold", color=INK, x=0.05, ha="left", y=0.97)
        if subtitle:
            ax.set_title(subtitle, fontsize=10, color=MUTE, loc="left")
        fig.text(0.98, 0.02, "SocioEconomicBot", ha="right", fontsize=8, color="#b0b4bb")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()

    def _fallback_bar(self, df, metric_col, title, subtitle):
        d = df.sort_values(metric_col, ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        ax.barh(d.iloc[:, 0].astype(str), d[metric_col], color="#2563eb")
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontweight="bold")
        if subtitle:
            fig.suptitle(subtitle, y=0.98, fontsize=9, color=MUTE)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()

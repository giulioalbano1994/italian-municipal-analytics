import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

# ==========================================================
#  ChartGenerator – modern, readable charts (v5)
# ==========================================================
# Curated qualitative palette: vivid but professional, colour-blind friendlyish.
PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706",
           "#7c3aed", "#0891b2", "#db2777", "#65a30d"]
INK = "#1f2937"       # near-black text
MUTE = "#6b7280"      # grey subtitle/footer
GRID = "#e5e7eb"      # light grid


def _fmt_compact(v: float) -> str:
    """1234567 -> '1,2 Mln'; 12000 -> '12 mila'; keeps small numbers plain (IT)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    a = abs(v)
    if a >= 1_000_000_000:
        s = f"{v/1_000_000_000:.1f} Mld"
    elif a >= 1_000_000:
        s = f"{v/1_000_000:.1f} Mln"
    elif a >= 1_000:
        s = f"{v/1_000:.1f} mila"
    elif a >= 1:
        s = f"{v:.0f}"
    else:
        s = f"{v:.2f}"
    return s.replace(".", ",")


class ChartGenerator:
    def __init__(self):
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.linewidth": 1.0,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "axes.labelcolor": MUTE,
            "text.color": INK,
            "xtick.color": MUTE,
            "ytick.color": MUTE,
            "legend.fontsize": 10,
            "font.size": 11,
            "font.family": "DejaVu Sans",
        })
        self.palette = PALETTE

    # ------------------------------------------------------
    def generate_chart(self, df: pd.DataFrame, chart_type: str, title: str,
                       xlabel: str, ylabel: str, subtitle: str = "") -> bytes:
        if df is None or df.empty:
            raise ValueError("DataFrame vuoto: impossibile generare il grafico.")

        x_col = next((c for c in ["anno", "comune", "provincia", "regione", "regione_mef", "metrica"]
                      if c in df.columns), df.columns[0])
        # Only numeric columns can be plotted as series — never the string 'comune'.
        y_cols = [c for c in df.columns if c != x_col and pd.api.types.is_numeric_dtype(df[c])]
        if not y_cols:
            raise ValueError("Nessuna colonna numerica da rappresentare.")

        if x_col == "anno":
            df = df.sort_values(by=x_col)
        df_plot = df.set_index(x_col)[y_cols]

        chart_type = (chart_type or "line").lower()
        # Categorical x (comune/regione/metrica) can't be a line → force bars.
        if chart_type in ("line", "lines") and x_col != "anno":
            chart_type = "bar"
        is_pct = "%" in (ylabel or "") or any("_pct" in c or "ratio" in c for c in y_cols)

        n_points, n_series = len(df_plot), len(y_cols)
        width = max(7, min(13, 6 + 0.28 * n_points))
        height = max(4.5, min(8, 4.6 + 0.4 * n_series))
        fig, ax = plt.subplots(figsize=(width, height), dpi=200)

        colors = (self.palette * ((n_series // len(self.palette)) + 1))[:n_series]

        # -------------------- chart types --------------------
        if chart_type in ("line", "lines"):
            for i, c in enumerate(y_cols):
                ax.plot(df_plot.index, df_plot[c], color=colors[i], linewidth=2.6,
                        marker="o", markersize=6, markerfacecolor="white",
                        markeredgecolor=colors[i], markeredgewidth=1.8, label=str(c))
            ax.margins(x=0.02)

        elif chart_type == "barh":  # ranking-friendly (leggibile per nomi comuni)
            df_plot = df_plot.sort_values(y_cols[0])
            ax.barh(df_plot.index.astype(str), df_plot[y_cols[0]], color=colors[0])
            for i, v in enumerate(df_plot[y_cols[0]]):
                ax.text(v, i, "  " + _fmt_compact(v), va="center", ha="left",
                        fontsize=9, color=INK)
            ax.margins(x=0.12)

        elif chart_type == "bar":
            df_plot.plot(kind="bar", ax=ax, color=colors, edgecolor="none", width=0.8, legend=False)
            if n_series == 1:  # value labels only when not too crowded
                for i, v in enumerate(df_plot[y_cols[0]]):
                    ax.text(i, v, _fmt_compact(v), ha="center", va="bottom", fontsize=9, color=INK)

        elif chart_type == "pie":
            if n_series != 1:
                raise ValueError("Il grafico a torta richiede una sola metrica.")
            ax.pie(df_plot.iloc[:, 0].clip(lower=0), labels=df_plot.index.astype(str),
                   autopct="%1.1f%%", colors=self.palette, startangle=90,
                   wedgeprops={"edgecolor": "white", "linewidth": 1.5})
            ax.set_ylabel("")
        else:
            for i, c in enumerate(y_cols):
                ax.plot(df_plot.index, df_plot[c], color=colors[i], linewidth=2.6,
                        marker="o", markersize=6, label=str(c))

        # -------------------- cosmetics --------------------
        fig.suptitle(title, fontsize=16, fontweight="bold", color=INK, x=0.02, ha="left", y=0.98)
        if subtitle:
            ax.set_title(subtitle, fontsize=10, color=MUTE, loc="left", pad=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if chart_type != "pie":
            ax.grid(axis="y", linestyle="-", linewidth=0.8, color=GRID, alpha=0.9)
            ax.set_axisbelow(True)
            # y formatting
            if is_pct:
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
            elif chart_type != "barh":
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _fmt_compact(v)))
            else:
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _fmt_compact(v)))

        if n_series > 1 and chart_type not in ("pie", "barh"):
            ax.legend(loc="best", frameon=False, ncol=min(n_series, 3))

        if chart_type in ("bar",) or x_col in ("comune", "provincia", "regione", "regione_mef"):
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

        fig.text(0.98, 0.01, "SocioEconomicBot", ha="right", va="bottom",
                 fontsize=8, color="#b0b4bb")

        buf = io.BytesIO()
        fig.tight_layout(rect=(0, 0.02, 1, 0.95))
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()

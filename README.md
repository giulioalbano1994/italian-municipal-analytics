# Italian Municipal Analytics Bot

A Telegram bot that answers plain-language questions — in **Italian or English** —
about the socio-economic profile of Italian municipalities, and replies with a
chart or map plus a short written analysis.

> _"Reddito medio Bari e Napoli nel tempo"_ · _"Classifica 10 comuni più ricchi"_ ·
> _"Laureati per regione"_ · _"/map reddito medio"_

Author: **Giulio Albano** — University of Bari (UNIBA), PhD in Economics and Finance
of Public Administrations.

---

## What it does

- **Natural language, two languages.** Ask a question; the bot detects the indicator,
  the municipalities (or regions/provinces), the time window, and the intent.
- **Real column mapping.** The full variable dictionary (name + description + synonyms)
  is fed to the LLM, and a resolver maps every requested term to an actual dataset
  column — so almost all 80+ variables are answerable, not just a hardcoded few.
- **The system picks the chart.** Chart type is chosen from the *shape* of the data,
  not guessed by the model: time → line, few categories → bars, many/long labels →
  horizontal bars, ranking → horizontal bars.
- **Comparison.** `Reddito medio Bari e Napoli nel tempo` draws one line per city.
- **Rankings.** `Classifica dei 10 comuni più ricchi`, top/bottom N, by municipality,
  province, or region.
- **Regional maps.** Choropleth of Italy by region — rendered from a GeoJSON with
  pure matplotlib (no geopandas dependency).
- **Random mode.** `/plot` and `/map` with no text generate a random, sensible
  chart/map — a one-tap tour of what the bot can do.
- **Written analysis.** An LLM commentary (headline + three insights + bottom line,
  Italian, Telegram-formatted) accompanies each answer; a numeric summary is used
  when no LLM key is set.
- **Guardrails.** Per-user rate limiting, nonsense filtering, and a lightweight
  classifier that routes help/info/offensive messages.

## Data

A single local table of ~196k rows × ~80 columns covering Italian municipalities
across years, assembled from official sources:

| Domain | Examples | Source |
|---|---|---|
| Income & tax | average income, taxable income, pensions, taxpayers | MEF |
| Population & migration | resident population, migration balance | ISTAT |
| Education | graduates (total / women / men) | MIUR |
| Inequality | Gini index | derived |
| Firms | active / registered firms, patents | Infocamere |
| Territory | region, province, NUTS3, capoluogo flag | ISTAT / Eurostat |

The variable dictionary lives in `resources/dizionario_variabili.csv`
(name, source, description, synonyms). Derived metrics (e.g. `reddito_medio`,
`laureati_pct`) are computed at load time.

## Architecture

```
main.py                      Telegram app: commands, menus, random mode, orchestration
modules/
  llm_processor.py           NL request → structured parameters; LLM commentary
  catalog.py                 variable dictionary → prompt block + synonym resolver
  data_query.py              filtering, long→wide reshaping, ranking, chart-type choice
  chart_generator.py         Matplotlib charts (line / bar / barh / pie)
  map_generator.py           regional choropleth from GeoJSON (no geopandas)
  classifier.py              lightweight intent gate (data / help / info / nonsense)
resources/
  df_ridotto_bot.csv         the municipal dataset (local, git-ignored)
  dizionario_variabili.csv   variable dictionary
  geo/regioni.geojson        Italian regions boundaries (local, git-ignored)
```

**Request lifecycle:** message → classifier → LLM parse (indicator + places + period
+ intent) → resolve metrics to real columns → query & reshape → choose chart / build
map → render → LLM commentary → reply.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

Configuration via `.env` (in the project root):

| Variable | Required | Purpose |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | ✅ | BotFather token |
| `OPENAI_API_KEY` | optional | OpenAI (`sk-…`) — enables LLM parsing & commentary |
| `IT_REGIONI_GEOJSON` | optional | Path to the regions GeoJSON (defaults to `resources/geo/regioni.geojson`) |

Without an OpenAI key the processor falls back to Gemini (if configured) or to a
deterministic path. The real `.env`, the dataset, and the GeoJSON are git-ignored —
never commit tokens or data.

## Run

```bash
python main.py
```

Only one polling instance may run at a time.

## Usage

```
Reddito medio Bari e Napoli nel tempo      # comparison → two lines
Classifica dei 10 comuni più ricchi        # ranking
Laureati per regione                       # aggregation
/map reddito medio                         # regional choropleth
/plot                                      # random chart
/map                                       # random map
```

## Robustness & performance

- **No per-query DataFrame copy** — filters run directly on the loaded frame; a
  lowercase municipality key is precomputed once. The dataset is read a single time.
- **Long→wide reshaping** before charting, so string columns are never plotted as a
  data series (the classic phantom "= 0" line).
- **LLM parses are cached** per normalized question.
- **Choropleth without geopandas** — GeoJSON parsed with `json`, drawn with
  matplotlib; regions joined to data by name-normalization (accents, hyphens, and
  bilingual ISTAT names handled).
- **Telegram-safe text** — commentary avoids Markdown headers and falls back to
  plain text if a message fails to parse.

## Limits & notes

- Maps are at **regional** level (a regions GeoJSON is bundled locally); province-level
  maps would need a provinces GeoJSON.
- LLM commentary via OpenAI/Gemini incurs cost; validate and monitor your key.
- Data availability follows the underlying official sources.

## Attribution

Data: ISTAT, MEF, MIUR, Infocamere, Eurostat. Region boundaries:
[openpolis / geojson-italy](https://github.com/openpolis/geojson-italy). Built with
`python-telegram-bot`, pandas, and matplotlib.

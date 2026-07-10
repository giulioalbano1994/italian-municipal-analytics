import os
import sys
import logging
from io import BytesIO

# Windows console is cp1252 → emoji in print/log crash. Force UTF-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque
import re
import pandas as pd

from telegram import (
    Update, ReplyKeyboardMarkup, KeyboardButton,
    InlineKeyboardButton, InlineKeyboardMarkup, InlineQueryResultArticle,
    InputTextMessageContent
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes,
    CallbackQueryHandler, InlineQueryHandler
)
from dotenv import load_dotenv

# --- Modules (local package) ---
from modules import (
    LLMProcessor, QueryType, ChartType, QueryParameters,
    ChartGenerator
)
from modules.data_query import DataFrameManager
from modules.classifier import Classifier
from modules.map_generator import MapGenerator

# ---------- Config ----------
load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

LOG_FILE = Path(os.getenv("BOT_LOG_FILE", "conversazioni_bot.csv"))

# Facoltativo: teniamo il testo legacy ma NON lo usiamo più
HELP_TEXT = (
    "📊 **Come usare il bot**\n\n"
    "Scrivimi una domanda sui dati socio-economici comunali.\n\n"
    "⚙️ Comandi: /start /help /info /plot /map\n"
)

INFO_TEXT_BASE = (
    "ℹ️ **Informazioni sul Bot**\n\n"
    "👨‍💻 Autore: *Giulio Albano* (Univ. Bari, tirocinio Banca d'Italia)\n"
    "📚 Fonti: *ISTAT, MEF, MIUR, Infocamere, Eurostat*\n\n"
    "🔎 Il bot normalizza i dati, calcola metriche derivate "
    "(pro capite, percentuali, quote, crescite) e genera grafici leggibili. "
    "Perfetto per confronti tra comuni, serie storiche e analisi distributive."
)

# Limiti e blocchi
MAX_REQUESTS_PER_MINUTE = 5
MAX_NONSENSE = 5
BLOCK_DURATION = timedelta(minutes=30)
user_requests = defaultdict(deque)
user_nonsense = defaultdict(int)
user_blocked = {}

# ---------- Utility ----------

def log_message(user, message_text, message_type, direction="IN", query_type=None, comuni=None, metrics=None):
    clean_text = message_text.encode("utf-8", "ignore").decode("utf-8") if message_text else ""
    clean_text = clean_text.replace("\n", " ").replace("\r", " ").replace('"', "'")
    rec = {
        "timestamp": pd.Timestamp.now(),
        "user_id": user.id if user else "",
        "username": (user.username or user.first_name) if user else "",
        "first_name": user.first_name if user else "",
        "last_name": (user.last_name or "") if user else "",
        "direction": direction,
        "message_type": message_type,
        "message_text": clean_text,
        "query_type": query_type or "",
        "comuni": ",".join(comuni) if comuni else "",
        "metrics": ",".join(metrics) if metrics else "",
        "character_count": len(clean_text),
        "word_count": len(clean_text.split()) if clean_text else 0
    }
    df = pd.DataFrame([rec])
    try:
        if LOG_FILE.exists():
            df.to_csv(LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig", sep=";")
        else:
            df.to_csv(LOG_FILE, mode="w", header=True, index=False, encoding="utf-8-sig", sep=";")
    except Exception as e:
        logger.error(f"log save error: {e}")

def is_user_blocked(user_id: int) -> bool:
    if user_id in user_blocked:
        if datetime.now() < user_blocked[user_id]:
            return True
        else:
            del user_blocked[user_id]
            user_nonsense[user_id] = 0
    return False

def register_request(user_id: int) -> bool:
    now = datetime.now()
    dq = user_requests[user_id]
    while dq and (now - dq[0]).seconds > 60:
        dq.popleft()
    dq.append(now)
    if len(dq) > MAX_REQUESTS_PER_MINUTE:
        user_blocked[user_id] = now + BLOCK_DURATION
        return False
    return True

def is_nonsense_message(message: str) -> bool:
    s = (message or "").strip().lower()
    if len(s) < 2:
        return True
    if re.match(r"^[^\w\s]+$", s) or re.match(r"^\d+$", s):
        return True
    if re.match(r"^(.)\1{4,}$", s):
        return True
    if re.match(r"^[a-z]{10,}$", s) and not any(v in s for v in "aeiou"):
        return True
    return False

# ---------- Bot Class ----------

class SocioEconomicBot:
    def __init__(self, token: str):
        self.token = token
        self.application = None
        self.df_manager = DataFrameManager(data_dir=os.getenv("DATA_DIR", r"resources"))
        self.chart_generator = ChartGenerator()
        self.map_generator = MapGenerator()
        self.report_generator = None  # PDF disattivati: tutto in app

        # Esempi brevi per callback (evita Button_data_invalid)
        self.examples_map = {
            "ex1": "Popolazione Milano e Roma nel tempo",
            "ex2": "Reddito medio Torino nel tempo",
            "ex3": "Gini index Bari e Palermo nel tempo",
            "ex4": "Laureati Firenze e Bologna (ultimo anno)",
        }

        openai_key = os.getenv("OPENAI_API_KEY")
        self.classifier = Classifier(openai_key) if openai_key else None
        # Sempre istanziato: LLMProcessor gestisce il fallback locale se la chiave manca
        self.llm_processor = LLMProcessor(openai_key or "")



        self.main_keyboard = ReplyKeyboardMarkup(
            [
                [KeyboardButton("/start"), KeyboardButton("/help")],
                [KeyboardButton("/info"), KeyboardButton("/plot")],
                [KeyboardButton("/map")],
            ],
            resize_keyboard=True,
            is_persistent=True,
        )

    # ---------- Setup ----------
    def setup(self):
        self.application = Application.builder().token(self.token).build()
        try:
            self.df_manager.load_data()
            # Give the LLM the real columns + comuni (also avoids a 2nd 90MB read)
            self.llm_processor.set_context(
                self.df_manager.available_variables(9999),
                self.df_manager.comuni_list(),
            )
        except Exception as e:
            logger.error(f"Errore caricamento dati: {e}")
        self._register_handlers()

    def _register_handlers(self):
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("info", self.info_command))
        self.application.add_handler(CommandHandler("plot", self.plot_command))
        self.application.add_handler(CommandHandler("map", self.map_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_question))
        # Inline buttons handlers (examples + variables)
        self.application.add_handler(CallbackQueryHandler(self.handle_example_callback, pattern=r"^EXAMPLE:.+"))
        self.application.add_handler(CallbackQueryHandler(self.handle_show_vars_callback, pattern=r"^SHOW_VARS$"))
        # Inline mode
        self.application.add_handler(InlineQueryHandler(self.handle_inline_query))

    # ---------- Commands ----------
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        msg = (
            f"Ciao {user.first_name}! 👋\n\n"
            "Sono il tuo bot sui dati socio-economici 🇮🇹\n\n"
            "Esempi pronti:\n"
            "• Popolazione Bari e Napoli nel tempo\n"
            "• Reddito medio a Milano 2010–2020\n"
            "• Quota pensionati Roma e Firenze\n\n"
            "Scrivi la tua richiesta o usa /help."
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=self.main_keyboard)
        log_message(user, msg, "RESPONSE", "OUT", query_type="START")

        # Inline keyboard di esempi sotto il benvenuto
        rows = [[InlineKeyboardButton(text=label, callback_data=f"EXAMPLE:{code}")]
                for (code, label) in self.examples_map.items()]
        inline_kb = InlineKeyboardMarkup(rows)
        await update.message.reply_text(
            "✨ Esempi rapidi (clicca per generare il grafico):",
            reply_markup=inline_kb
        )

    # ---------- /help ----------
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Mostra una guida compatta e una tastiera di esempi cliccabili (come /start).
        """
        msg = (
            "📘 *Come usare il bot*\n\n"
            "Puoi chiedermi dati, confronti o serie storiche sui comuni italiani 🇮🇹\n\n"
            "✨ *Esempi rapidi (clicca per generare il grafico):*"
        )
        rows = [
            [InlineKeyboardButton(text=label, callback_data=f"EXAMPLE:{code}")]
            for code, label in self.examples_map.items()
        ]
        inline_kb = InlineKeyboardMarkup(rows)
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=inline_kb)

    async def info_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        # Meta info dal dataset
        meta = self.df_manager.dataset_meta()
        coverage = meta.get("coverage_str", "—")
        latest_year = meta.get("latest_year", "—")
        sources = ", ".join(sorted(meta.get("sources", []))) or "—"

        msg = (
            INFO_TEXT_BASE
            + f"\n\n📅 Ultimo anno disponibile: *{latest_year}*"
            + f"\n📄 Coverage righe: *{coverage}*"
            + f"\n📁 Fonti attive nel dataset: *{sources}*"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=self.main_keyboard)
        log_message(user, msg, "RESPONSE", "OUT", query_type="INFO")

    # ---------- Callback: Esempi cliccabili ----------
    async def handle_example_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        q = update.callback_query
        await q.answer()
        data = q.data or ""
        if data.startswith("EXAMPLE:"):
            code = data.split(":", 1)[1]
            query = self.examples_map.get(code, code)
            if query:
                await self.plot_command(update, context, user_input=query)

    # ---------- Callback: Mostra variabili ----------
    async def handle_show_vars_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        q = update.callback_query
        await q.answer()
        user = update.effective_user
        try:
            vars = self.llm_processor.available_variables(limit=9999) if self.llm_processor else []
        except Exception:
            vars = []

        if not vars:
            await q.message.reply_text("Nessuna variabile disponibile al momento.")
            return

        chunk, total_sent, max_chars, current_len = [], 0, 3500, 0
        for v in vars:
            piece = ("`" + v + "`")
            if current_len + len(piece) + (2 if chunk else 0) > max_chars:
                await q.message.reply_text("🔑 Variabili:\n" + ", ".join(chunk), parse_mode=ParseMode.MARKDOWN)
                total_sent += 1
                chunk, current_len = [piece], len(piece)
            else:
                if chunk:
                    chunk.append(piece)
                    current_len += len(piece) + 2
                else:
                    chunk, current_len = [piece], len(piece)
        if chunk:
            await q.message.reply_text("🔑 Variabili:\n" + ", ".join(chunk), parse_mode=ParseMode.MARKDOWN)
            total_sent += 1
        log_message(user, f"vars_sent={total_sent}", "RESPONSE", "OUT", query_type="VARS")

    # ---------- Inline Mode ----------
    async def handle_inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Basic inline mode: echo a parsed summary + prompt to open chat for full chart."""
        query = (update.inline_query.query or "").strip()
        if not query:
            return
        title = f"Analizza: {query}"
        desc = "Tocca per inviare la richiesta. Apri la chat con il bot per ottenere grafici e mappe."
        content = InputTextMessageContent(f"Richiesta inviata al bot: *{query}*\nApri la chat del bot per il grafico.")
        await update.inline_query.answer(
            results=[InlineQueryResultArticle(
                id="1", title=title, description=desc, input_message_content=content, thumbnail_url=None
            )],
            cache_time=5, is_personal=True
        )

    # ---------- Gestione messaggi liberi ----------
    async def handle_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        text = (update.message.text or "").strip()

        try:
            if self.classifier:
                res = self.classifier.classify(text)
                cat = res.get('category')
                logger.info(f"classifier.category={cat}")
                if cat == 'help_request':
                    await self.help_command(update, context)
                    return
                if cat == 'info_request':
                    await self.info_command(update, context)
                    return
                if cat == 'offensive':
                    await update.message.reply_text("⚠️ Per favore usa un linguaggio rispettoso.")
                    return
                # Non bloccare se l'euristica dice 'nonsense' ma il classificatore vede 'data_request'
                if cat == 'nonsense':
                    await update.message.reply_text("⚠️ Messaggio non riconosciuto. Usa /help per esempi.")
                    return
        except Exception:
            pass

        if is_user_blocked(user.id):
            await update.message.reply_text("⚠️ Sei temporaneamente bloccato. Riprova tra 30 minuti.")
            return
        if not register_request(user.id):
            await update.message.reply_text("⚠️ Troppe richieste. Riprova tra 30 minuti.")
            return

        # Euristica nonsense: la saltiamo se il testo contiene parole chiave data
        nonsense = is_nonsense_message(text)
        logger.info(f"is_nonsense={nonsense}")
        if nonsense:
            user_nonsense[user.id] += 1
            if user_nonsense[user.id] >= MAX_NONSENSE:
                user_blocked[user.id] = datetime.now() + BLOCK_DURATION
                await update.message.reply_text("⚠️ Messaggi non validi. Riprova tra 30 minuti.")
            else:
                await update.message.reply_text("⚠️ Messaggio non riconosciuto. Usa /help per esempi.")
            return

        await self.plot_command(update, context, user_input=text)

    # ---------- /plot ----------
    async def plot_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_input: str = None):
        # message object (supporta callback)
        msg_obj = update.message or (update.callback_query.message if getattr(update, "callback_query", None) else None)
        user = update.effective_user
        text = user_input or ("" if not context.args else " ".join(context.args))
        if not text:
            await msg_obj.reply_text("❓ Usa `/plot popolazione Milano` o simili.", parse_mode=ParseMode.MARKDOWN)
            return

        processing_msg = await msg_obj.reply_text("🔄 Sto elaborando la tua richiesta...")

        try:
            if self.llm_processor is None:
                await processing_msg.edit_text("❌ LLM non configurato. Aggiungi OPENAI_API_KEY nel .env")
                return

            params = self.llm_processor.process_request(text)
            logger.info(f"🔍 Params: {params}")

            if params.query_type == QueryType.RANKING:
                df, xlabel, ylabel, meta = self.df_manager.query_ranking(params)
                params.chart_type = ChartType.BARH
            else:
                df, xlabel, ylabel, meta = self.df_manager.query_data(params)
                tl = text.lower()
                if any(k in tl for k in ['andamento', 'nel tempo', 'storico', 'evoluzione', 'trend']):
                    params.chart_type = ChartType.LINE

            if df is None or df.empty:
                period_label = self._period_label(df, params)
                detail = f"\n\n🔎 Parametri: comuni={params.comuni or '—'}, metrics={params.metrics or '—'}, periodo={period_label or '—'}"
                hint = "\n💡 Prova: `Popolazione Bari e Napoli nel tempo` oppure usa /help"
                await processing_msg.edit_text("❌ Nessun dato trovato." + detail + hint, parse_mode=ParseMode.MARKDOWN)
                return

            # Title & subtitle
            metrics_label = ' / '.join(params.metrics or [])
            if params.query_type == QueryType.RANKING:
                n = params.top_n or 10
                order = "meno" if params.ascending else "più"
                lvl = meta.get("level", "comune")
                title = f"Classifica {metrics_label} • {n} {lvl} {order} • {meta.get('rank_year', '')}"
            else:
                period_label = self._period_label(df, params)
                comuni_label = ", ".join(params.comuni) if params.comuni else "(tutti i comuni)"
                title = " • ".join([p for p in [metrics_label, comuni_label, period_label] if p])

            subtitle = self._subtitle_from_meta(meta)

            img = self.chart_generator.generate_chart(
                df,
                chart_type=(getattr(params.chart_type, 'value', params.chart_type)),
                title=title, subtitle=subtitle, xlabel=xlabel, ylabel=ylabel
            )

            # Optional: quick commentary (silenzioso se LLM non disponibile)
            comment = ""
            try:
                if self.llm_processor:
                    comment = self.llm_processor.generate_commentary(df, params)
            except Exception:
                comment = ""

            await msg_obj.reply_photo(BytesIO(img), caption=title, reply_markup=self.main_keyboard)

            if comment:
                await msg_obj.reply_text(comment)

            try:
                await processing_msg.delete()
            except Exception:
                pass

        except Exception as e:
            logger.exception("Errore nella generazione del grafico")
            await processing_msg.edit_text("❌ Errore durante l'elaborazione. Riprova.")
            log_message(user, str(e), "ERROR", "OUT")

    def _period_label(self, df: pd.DataFrame, params: QueryParameters) -> str:
        if params.anno:
            return str(params.anno)
        if params.start_year and params.end_year:
            return f"{params.start_year}-{params.end_year}"
        if df is not None and "anno" in df.columns and df["anno"].notna().any():
            ymin = int(pd.to_numeric(df["anno"], errors="coerce").dropna().min())
            ymax = int(pd.to_numeric(df["anno"], errors="coerce").dropna().max())
            return f"{ymin}-{ymax}" if ymin != ymax else str(ymin)
        return ""

    def _subtitle_from_meta(self, meta: dict) -> str:
        src = ", ".join(sorted(meta.get("sources", []))) if meta else ""
        latest = meta.get("latest_year") if meta else None
        coverage = meta.get("coverage_str") if meta else ""
        bits = []
        if src: bits.append(f"Fonte: {src}")
        if latest: bits.append(f"Ultimo anno: {latest}")
        if coverage: bits.append(f"Coverage: {coverage}")
        return " • ".join(bits)

    def _suggest_alternatives(self, text: str) -> tuple[str, str]:
        t = (text or "").strip()
        tokens = [w for w in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", t) if len(w) > 2]
        city = None
        for w in tokens:
            if w[:1].isupper():
                city = w
                break
        if not city and tokens:
            city = tokens[-1].capitalize()
        city = city or "Torino"
        return (f"Reddito medio {city} nel tempo", f"Popolazione {city} 2015–2023")

    # ---------- /map ----------
    async def map_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg_obj = update.message
        text = " ".join(context.args) if context.args else ""
        if not text:
            await msg_obj.reply_text("🗺️ Sintassi: `/map <metrica> [anno]` es. `/map average_income 2023`", parse_mode=ParseMode.MARKDOWN)
            return
        try:
            anno = None
            tokens = text.split()
            metric = tokens[0]
            if len(tokens) > 1 and tokens[1].isdigit():
                anno = int(tokens[1])
            params = QueryParameters(
                query_type=QueryType.SINGLE_COMUNE, chart_type=ChartType.BAR,
                metrics=[metric], anno=anno
            )
            # Otteniamo ultimo anno aggregato a livello regionale (o comunale se manca regioni)
            df, xlabel, ylabel, meta = self.df_manager.query_data_for_map(params)
            if df is None or df.empty:
                await msg_obj.reply_text("❌ Nessun dato mappabile trovato per quella metrica/anno.")
                return
            title = f"Mappa • {metric} • {anno or meta.get('latest_year','ultimo anno')}"
            subtitle = self._subtitle_from_meta(meta)
            png = self.map_generator.generate_choropleth(
                df, metric_col=metric, level=meta.get("map_level", "regione"),
                title=title, subtitle=subtitle
            )
            await msg_obj.reply_photo(BytesIO(png), caption=title, reply_markup=self.main_keyboard)
        except Exception:
            logger.exception("Errore mappa")
            await msg_obj.reply_text("❌ Errore durante la generazione della mappa.")

    # ---------- Run ----------
    def run(self):
        if not self.application:
            self.setup()
        logger.info("Bot in esecuzione...")
        self.application.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
    if not token:
        print("❌ TOKEN mancante. Imposta TELEGRAM_BOT_TOKEN o TELEGRAM_TOKEN (oppure verifica il file .env)")
        return

    bot = SocioEconomicBot(token)
    try:
        print("✅ Bot avviato e in ascolto su Telegram (Ctrl+C per fermarlo)")
        bot.run()
    except KeyboardInterrupt:
        print("🛑 Arresto richiesto dall'utente.")
    except Exception as e:
        print(f"❌ Errore critico: {e}")


if __name__ == "__main__":
    print("🚀 Avvio del bot...")
    main()
    print("✅ Bot in esecuzione (usa Ctrl+C per fermarlo)")

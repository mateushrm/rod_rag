import os
from collections import defaultdict
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes
)

from consultar_rod import buscar_faq, responder_rag

TOKEN = os.getenv("TELEGRAM_TOKEN")

historicos = defaultdict(list)

# Mensagem de boas-vindas
MENSAGEM_INICIO = (
    "Olá! Eu sou o RodBot, seu assistente sobre o ROD do IFCE campus Tianguá.\n\n"
    "Escolha uma categoria ou digite sua dúvida diretamente:"
)

# Saudações
SAUDACOES = {
    "oi", "olá", "ola", "hello", "hi", "hey",
    "bom dia", "boa tarde", "boa noite",
    "tudo bem", "boa", "e aí", "eai", "eaí"
}

# Despedidas
DESPEDIDAS = {
    "tchau", "bye", "até mais", "ate mais", "obrigado", "obrigada",
    "valeu", "vlw", "flw", "falou", "até", "ate", "adeus", "encerrar"
}

# Perguntas sugeridas por categoria
PERGUNTAS_CATEGORIA = {
    "matricula": {
        "titulo": "Matrícula",
        "perguntas": [
            "Como faço a renovação de matrícula?",
            "Como solicitar mudança de turno?",
        ]
    },
    "avaliacoes": {
        "titulo": "Avaliações e Notas",
        "perguntas": [
            "Qual a média para passar direto?",
            "Como funciona a prova final?",
            "Como solicitar segunda chamada?",
        ]
    },
    "faltas": {
        "titulo": "Faltas e Frequência",
        "perguntas": [
            "Quantas faltas posso ter?",
            "Como justificar minhas faltas?",
        ]
    },
    "trancamento": {
        "titulo": "Trancamento",
        "perguntas": [
            "Como faço para trancar meu curso?",
            "Por quanto tempo posso ficar trancado?",
            "Posso trancar no primeiro semestre?",
            "Posso trancar só uma disciplina?",
        ]
    },
    "sei": {
        "titulo": "SEI e Protocolos",
        "perguntas": [
            "Como acesso o SEI pela primeira vez?",
            "Como abrir um processo no SEI?",
            "Como acompanhar meu processo?",
        ]
    },
    "direitos": {
        "titulo": "Direitos Discentes",
        "perguntas": [
            "O que é o Regime de Exercícios Domiciliares?",
            "Tenho direito ao uso do nome social?",
            "O que é progressão parcial?",
        ]
    }
}


def menu_principal():
    botoes = [
        [
            InlineKeyboardButton("Matrícula", callback_data="cat_matricula"),
            InlineKeyboardButton("Avaliações e Notas", callback_data="cat_avaliacoes"),
        ],
        [
            InlineKeyboardButton("Faltas e Frequência", callback_data="cat_faltas"),
            InlineKeyboardButton("Trancamento", callback_data="cat_trancamento"),
        ],
        [
            InlineKeyboardButton("SEI e Protocolos", callback_data="cat_sei"),
            InlineKeyboardButton("Direitos Discentes", callback_data="cat_direitos"),
        ]
    ]
    return InlineKeyboardMarkup(botoes)


def menu_categoria(categoria: str):
    perguntas = PERGUNTAS_CATEGORIA[categoria]["perguntas"]
    botoes = [[InlineKeyboardButton(p, callback_data=f"perg_{p}")] for p in perguntas]
    botoes.append([InlineKeyboardButton("Voltar ao menu", callback_data="menu")])
    return InlineKeyboardMarkup(botoes)


def menu_voltar():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Voltar ao menu", callback_data="menu")]
    ])


def buscar_resposta(pergunta: str, chat_id: int):
    historico = historicos[chat_id]

    # FAQ usa pergunta limpa
    resposta = buscar_faq(pergunta)

    # RAG usa pergunta com contexto histórico
    if not resposta:
        if historico:
            pergunta_com_contexto = f"{' '.join(historico[-2:])} {pergunta}"
        else:
            pergunta_com_contexto = pergunta
        resposta = responder_rag(pergunta_com_contexto)

    historico.append(pergunta)
    if len(historico) > 5:
        historico.pop(0)

    if not resposta or not resposta.strip():
        resposta = "Não encontrei essa informação nos documentos disponíveis. Consulte a coordenadoria do seu curso ou a secretaria do campus."

    # Proteção contra mensagens longas (limite Telegram: 4096 chars)
    if len(resposta) > 4000:
        resposta = resposta[:4000] + "...\n\nConsulte a coordenadoria para mais detalhes."

    return resposta


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    historicos[update.effective_chat.id].clear()
    await update.message.reply_text(
        MENSAGEM_INICIO,
        reply_markup=menu_principal()
    )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = query.message.chat_id

    # Voltou ao menu principal
    if data == "menu":
        await query.edit_message_text(
            MENSAGEM_INICIO,
            reply_markup=menu_principal()
        )

    # Selecionou uma categoria
    elif data.startswith("cat_"):
        categoria = data.replace("cat_", "")
        titulo = PERGUNTAS_CATEGORIA[categoria]["titulo"]
        await query.edit_message_text(
            f"Categoria: {titulo}\n\nEscolha uma pergunta ou digite sua dúvida:",
            reply_markup=menu_categoria(categoria)
        )

    # Selecionou uma pergunta
    elif data.startswith("perg_"):
        pergunta = data.replace("perg_", "")

        await context.bot.send_chat_action(
            chat_id=chat_id,
            action="typing"
        )

        resposta = buscar_resposta(pergunta, chat_id)

        await query.edit_message_text(
            f"Pergunta: {pergunta}\n\n{resposta}",
            reply_markup=menu_voltar()
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pergunta = update.message.text
    chat_id = update.effective_chat.id
    pergunta_lower = pergunta.strip().lower()

    # Saudações
    if pergunta_lower in SAUDACOES:
        historicos[chat_id].clear()
        await update.message.reply_text(
            MENSAGEM_INICIO,
            reply_markup=menu_principal()
        )
        return

    # Despedidas
    if pergunta_lower in DESPEDIDAS:
        historicos[chat_id].clear()
        await update.message.reply_text(
            "Até mais! Se tiver outras dúvidas sobre o ROD, é só chamar."
        )
        return

    await context.bot.send_chat_action(
        chat_id=chat_id,
        action="typing"
    )

    resposta = buscar_resposta(pergunta, chat_id)

    await update.message.reply_text(
        resposta,
        reply_markup=menu_voltar()
    )


app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(CallbackQueryHandler(handle_callback))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

app.run_polling()
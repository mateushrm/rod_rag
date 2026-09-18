from fastapi import FastAPI
from pydantic import BaseModel
from collections import defaultdict
from consultar_rod import buscar_faq, responder_rag, verificar_categoria

app = FastAPI()

#Modelo de dados para o Dialogflow
class DialogflowRequest(BaseModel):
    queryResult: dict
    session: str = "default"

# Memória de sessão
sessoes = defaultdict(list)


@app.post("/webhook")
async def webhook(req: DialogflowRequest):
    try:
        body = req.dict()
        pergunta = body["queryResult"]["queryText"]
        session_id = body.get("session", "default")

        print("PERGUNTA:", pergunta)
        print("SESSION:", session_id)

        historico = sessoes[session_id]

	# Verifica categoria
        resposta = verificar_categoria(pergunta)

        # FAQ usa só a pergunta atual
        if not resposta:
            resposta = buscar_faq(pergunta)

        # RAG usa contexto histórico para perguntas de acompanhamento
        if not resposta:
            if historico:
                pergunta_com_contexto = f"{' '.join(historico[-2:])} {pergunta}"
            else:
                pergunta_com_contexto = pergunta
            resposta = responder_rag(pergunta_com_contexto)

        # Salva no histórico
        historico.append(pergunta)
        if len(historico) > 5:
            historico.pop(0)

        if not resposta or not resposta.strip():
            resposta = "Não encontrei essa informação no ROD."

        print("RESPOSTA:", resposta)
        return {"fulfillmentText": resposta}

    except Exception as e:
        print("ERRO GERAL:", e)
        return {"fulfillmentText": "Erro interno no servidor. Tente novamente."}

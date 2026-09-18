import os
import re
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectorstore = Chroma(
    persist_directory=os.path.join(BASE_DIR, "db_rod"),
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 20}
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=300
)

prompt = PromptTemplate(
    template="""
Você é um assistente virtual do IFCE campus Tianguá, especializado em apoio acadêmico.

Você tem acesso aos seguintes documentos:
1. ROD - Regulamento da Organização Didática do IFCE
2. Manual de Peticionamento Eletrônico do SEI para Alunos
3. Cartilha do SEI para Servidores

REGRAS ESTRITAS:
- Responda APENAS com base no contexto fornecido abaixo.
- Se a resposta não estiver no contexto, diga exatamente: "Não encontrei essa informação nos documentos disponíveis. Consulte a coordenadoria do seu curso ou a secretaria do campus."
- Seja direto e objetivo. Máximo 3 parágrafos.
- Cite o artigo ou o documento de origem sempre que possível.
- NUNCA invente informações.
- Se a pergunta envolver um processo (como trancamento, aproveitamento, segunda chamada), explique tanto a regra (ROD) quanto como protocolar no SEI, se essa informação estiver disponível no contexto.

Contexto:
{context}

Pergunta:
{question}

Resposta:
""",
    input_variables=["context", "question"]
)

# FAQ
CATEGORIAS = {
    "matricula": [
        "Como faco a renovacao de matricula?",
        "Como solicitar mudanca de turno?",
    ],
    "avaliacoes e notas": [
        "Qual a media para passar direto?",
        "Como funciona a prova final?",
        "Como solicitar segunda chamada?",
    ],
    "faltas e frequencia": [
        "Quantas faltas posso ter?",
        "Como justificar minhas faltas?",
    ],
    "trancamento": [
        "Como faco para trancar meu curso?",
        "Por quanto tempo posso ficar trancado?",
        "Posso trancar no primeiro semestre?",
        "Posso trancar so uma disciplina?",
    ],
    "sei e protocolos": [
        "Como acesso o SEI pela primeira vez?",
        "Como abrir um processo no SEI?",
        "Como acompanhar meu processo?",
    ],
    "direitos discentes": [
        "O que e o Regime de Exercicios Domiciliares?",
        "Tenho direito ao uso do nome social?",
        "O que e progressao parcial?",
    ]
}


def verificar_categoria(pergunta):
    import unicodedata
    def normalizar(texto):
        return unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8').lower().strip()
    
    pergunta_norm = normalizar(pergunta)
    for categoria, perguntas in CATEGORIAS.items():
        if normalizar(categoria) == pergunta_norm:
            return "Escolha uma pergunta:\n\n" + "\n".join("- " + p for p in perguntas)
    return None

with open(os.path.join(BASE_DIR, "faq.json"), "r", encoding="utf-8") as f:
    _faq = json.load(f)
  
def buscar_faq(pergunta: str):
    stopwords = {"de", "a", "o", "e", "em", "no", "na", "os", "as",
                 "do", "da", "um", "uma", "que", "se", "para", "com",
                 "por", "ao", "à", "ou", "meu", "minha", "eu", "me",
                 "certo", "gostaria", "saber", "qual", "como", "posso",
                 "tenho", "quero", "fazer", "isso", "pra", "pro"}

    def stemmer(palavra):
        sufixos = ['ação', 'ções', 'ando', 'endo', 'indo',
                   'ado', 'ada', 'idos', 'idas', 'ar', 'er', 'ir']
        for sufixo in sufixos:
            if palavra.endswith(sufixo) and len(palavra) - len(sufixo) >= 4:
                return palavra[:-len(sufixo)]
        return palavra

    def tokenizar(texto):
        texto = re.sub(r'[^\w\s]', '', texto.lower())
        return set(
            stemmer(p) for p in texto.split()
            if len(p) > 3 and p not in stopwords
        )

    tokens_pergunta = tokenizar(pergunta)

    if not tokens_pergunta:
        return None

    melhor_item = None
    melhor_score = 0

    for item in _faq:
        tokens_faq = tokenizar(item["pergunta"])
        if not tokens_faq:
            continue
        intersecao = tokens_pergunta & tokens_faq
        score = len(intersecao) / len(tokens_pergunta | tokens_faq)

        if score > melhor_score:
            melhor_score = score
            melhor_item = item

    if melhor_score >= 0.07 and melhor_item:
        return melhor_item["resposta"]

    return None


def responder_rag(pergunta: str):
    try:
        docs = retriever.invoke(pergunta)

        if not docs:
            return "Não encontrei essa informação nos documentos disponíveis. Consulte a coordenadoria do seu curso."

        contexto = "\n\n".join([
            f"[Fonte: {doc.metadata.get('fonte', 'desconhecida')}]\n{doc.page_content}"
            for doc in docs
        ])

        resposta = llm.invoke(
            prompt.format(context=contexto, question=pergunta)
        )

        return resposta.content.strip()

    except Exception as e:
        print("ERRO NO RAG:", e)
        return "Não encontrei essa informação nos documentos disponíveis. Consulte a coordenadoria do seu curso."


def responder(pergunta: str):
    # 1. Verifica se é uma categoria
    resposta_categoria = verificar_categoria(pergunta)
    if resposta_categoria:
        return resposta_categoria

    # 2. FAQ
    resposta_faq = buscar_faq(pergunta)
    if resposta_faq:
        return resposta_faq
 
    # 3. RAG
    return responder_rag(pergunta)


if __name__ == "__main__":
    pergunta = input("Digite sua pergunta: ")
    print(responder(pergunta))

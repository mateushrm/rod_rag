import os
import re
import json
import unicodedata
from dotenv import load_dotenv
load_dotenv()
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG = False  # True = mostra origem, consultas e chunks


def log(*args):
    if DEBUG:
        print("[DEBUG]", *args)


def sem_acento(texto: str) -> str:
    return (
        unicodedata.normalize("NFD", texto)
        .encode("ascii", "ignore")
        .decode("utf-8")
        .lower()
        .strip()
    )


# ---------------- MODELOS E BANCO VETORIAL ----------------
# Mesma configuração usada na indexação (obrigatório)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    persist_directory=os.path.join(BASE_DIR, "db_rod"),
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8},
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=800,
)

prompt = PromptTemplate(
    template="""
Você é o RodBot, assistente virtual do IFCE campus Tianguá, que ajuda estudantes com dúvidas acadêmicas.

Você consulta estes documentos:
- ROD: Regulamento da Organização Didática do IFCE (regras acadêmicas)
- Manual de Peticionamento Eletrônico do SEI para Alunos (como protocolar pedidos)
- Cartilha do SEI para Servidores

COMO RESPONDER:
1. Use SOMENTE o contexto abaixo. Nunca complete com conhecimento próprio nem invente prazos, artigos, números ou documentos.
2. Fale diretamente com o estudante, em linguagem simples, sem dizer "de acordo com o contexto" ou "o trecho fornecido".
3. Comece pela resposta à pergunta, sem introdução.
4. Se a resposta for uma lista de documentos, requisitos ou etapas, use itens curtos ou passos numerados.
5. Cite a fonte no formato , usando o documento e a página indicados no início de cada trecho. Cite o número do artigo quando ele aparecer no texto.
6. Máximo de 3 parágrafos curtos ou uma lista objetiva.

SITUAÇÕES:
- Se o contexto responde à pergunta: responda.
- Se o contexto responde só em parte, ou traz o procedimento geral mas não o detalhe perguntado (por exemplo, um canal como e-mail): diga o que os documentos estabelecem e avise, numa frase, que o detalhe perguntado não consta neles. Nunca diga que algo é permitido se o contexto não afirma isso.
- Se o contexto não tem nada relacionado à pergunta, responda exatamente: "Não encontrei essa informação nos documentos disponíveis. Consulte a coordenadoria do seu curso ou a secretaria do campus."
- Se a pergunta for saudação ou conversa fora do tema acadêmico, responda brevemente e convide o estudante a perguntar sobre regras do ROD ou uso do SEI.

Contexto:
{context}

Pergunta do estudante:
{question}

Resposta:
""",
    input_variables=["context", "question"],
)

MSG_NAO_ENCONTRADO = (
    "Não encontrei essa informação nos documentos disponíveis. "
    "Consulte a coordenadoria do seu curso ou a secretaria do campus."
)

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
    ],
}


def verificar_categoria(pergunta: str):
    pergunta_norm = sem_acento(pergunta)
    for categoria, perguntas in CATEGORIAS.items():
        if sem_acento(categoria) == pergunta_norm:
            return "Escolha uma pergunta:\n\n" + "\n".join("- " + p for p in perguntas)
    return None


with open(os.path.join(BASE_DIR, "faq.json"), "r", encoding="utf-8") as f:
    _faq = json.load(f)

STOPWORDS = {
    "de", "a", "o", "e", "em", "no", "na", "os", "as", "do", "da", "um", "uma",
    "que", "se", "para", "com", "por", "ao", "ou", "meu", "minha", "eu", "me",
    "certo", "gostaria", "saber", "qual", "como", "posso", "tenho", "quero",
    "fazer", "isso", "pra", "pro",
}

SUFIXOS = ["acao", "coes", "ando", "endo", "indo",
           "ado", "ada", "idos", "idas", "ar", "er", "ir"]

FAQ_MIN_TOKENS_COMUNS = 2
FAQ_LIMIAR = 0.5


def stemmer(palavra: str) -> str:
    for sufixo in SUFIXOS:
        if palavra.endswith(sufixo) and len(palavra) - len(sufixo) >= 4:
            return palavra[:-len(sufixo)]
    return palavra


def tokenizar(texto: str) -> set:
    texto = re.sub(r"[^\w\s]", "", sem_acento(texto))
    return {
        stemmer(p) for p in texto.split()
        if len(p) > 3 and p not in STOPWORDS
    }


for _item in _faq:
    _item["_tokens"] = tokenizar(_item["pergunta"])


def buscar_faq(pergunta: str):
    tokens_pergunta = tokenizar(pergunta)
    if not tokens_pergunta:
        return None

    melhor_item = None
    melhor_score = 0

    for item in _faq:
        tokens_faq = item["_tokens"]
        if not tokens_faq:
            continue

        intersecao = tokens_pergunta & tokens_faq
        if len(intersecao) < FAQ_MIN_TOKENS_COMUNS:
            continue

        score = len(intersecao) / len(tokens_pergunta | tokens_faq)
        if score > melhor_score:
            melhor_score = score
            melhor_item = item

    log(f"FAQ melhor score: {melhor_score:.2f}",
        f"({melhor_item['pergunta']})" if melhor_item else "")

    if melhor_score >= FAQ_LIMIAR and melhor_item:
        return melhor_item["resposta"]

    return None

# MAPIEI TODAS AS SIGLAS DOS 3 DOCS 
SIGLAS = {
    "RED": "Regime de Exercícios Domiciliares",
    "ROD": "Regulamento da Organização Didática",
    "SEI": "Sistema Eletrônico de Informações",
    "PPE": "Progressão Parcial de Estudos",
    "PPC": "Projeto Pedagógico de Curso",
    "PUD": "Programa de Unidade Didática",
    "NDE": "Núcleo Docente Estruturante",
    "CAE": "Coordenadoria de Assuntos Estudantis",
    "CCA": "Coordenação de Controle Acadêmico",
    "CTP": "Coordenação Técnico-Pedagógica",
    "EAD": "Educação a Distância",
    "EJA": "Educação de Jovens e Adultos",
    "IRA": "Índice de Rendimento Acadêmico",
    "ENADE": "Exame Nacional de Desempenho de Estudantes",
    "ENEM": "Exame Nacional do Ensino Médio",
    "NUP": "Número Único de Protocolo",
    "PROEN": "Pró-Reitoria de Ensino",
    "CONSUP": "Conselho Superior",
}


def expandir_siglas(pergunta: str) -> str:
    for sigla, nome in SIGLAS.items():
        if nome.lower() not in pergunta.lower():
            pergunta = re.sub(
                rf"\b{sigla}\b", f"{nome} ({sigla})", pergunta, flags=re.IGNORECASE
            )
    return pergunta


def reescrever_consulta(pergunta: str) -> str:
    """Transforma a pergunta do aluno em termos formais do regulamento."""
    msg = (
        "Reescreva a pergunta abaixo como uma consulta de busca no Regulamento "
        "da Organização Didática do IFCE. Use termos formais e sinônimos "
        "(ex.: requerimento, atestado médico, laudo, prazo, comprovação, "
        "justificativa de faltas, abono), expanda siglas e responda só com "
        "a consulta, em uma linha.\n\n"
        f"Pergunta: {pergunta}"
    )
    return llm.invoke(msg).content.strip()


def buscar_docs(pergunta: str):
    consultas = [expandir_siglas(pergunta), reescrever_consulta(pergunta)]
    log("consultas:", consultas)

    vistos, docs = set(), []
    for c in consultas:
        for d in retriever.invoke(c):
            if d.page_content not in vistos:
                vistos.add(d.page_content)
                docs.append(d)
    return docs


def responder_rag(pergunta: str):
    try:
        docs = buscar_docs(pergunta)

        if not docs:
            return MSG_NAO_ENCONTRADO

        for d in docs:
            log("chunk:", d.page_content[:150].replace("\n", " "))

        contexto = "\n\n".join(
            f"[Fonte: {doc.metadata.get('fonte', 'desconhecida')}]\n{doc.page_content}"
            for doc in docs
        )

        resposta = llm.invoke(prompt.format(context=contexto, question=pergunta))
        return resposta.content.strip()

    except Exception as e:
        print("ERRO NO RAG:", e)
        return MSG_NAO_ENCONTRADO

def responder(pergunta: str):
    resp = verificar_categoria(pergunta)
    if resp:
        log("origem: categoria")
        return resp

    resp = buscar_faq(pergunta)
    if resp:
        log("origem: FAQ")
        return resp

    log("origem: RAG")
    return responder_rag(pergunta)


if __name__ == "__main__":
    print("RodBot pronto! Digite 'sair' para encerrar.\n")
    try:
        while True:
            pergunta = input("Digite sua pergunta: ").strip()

            if pergunta.lower() in ("sair", "exit", "quit"):
                print("Até logo!")
                break

            if not pergunta:
                continue

            print(responder(pergunta))
            print()
    except KeyboardInterrupt:
        print("\nAté logo!")
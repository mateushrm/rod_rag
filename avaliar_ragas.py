"""
Avaliação do RodBot usando o framework RAGAS
Execução: python avaliar_ragas.py

Instalação das dependências:
pip install ragas datasets openai langchain
"""

import json
import math
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from openai import OpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from langchain_openai import OpenAIEmbeddings
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# Importa o sistema RAG do RodBot
from consultar_rod import retriever, llm, prompt

# Juiz do RAGAS: modelo mais forte que o gerador (gpt-4o-mini), para evitar
# viés de autoavaliação
MODELO_JUIZ = "gpt-4o"

# ============================================================
# BANCO DE QUESTÕES (baseadas no ROD e SEI)
# ============================================================

questoes = [
    {
        "pergunta": "Qual é a frequência mínima exigida para aprovação em uma disciplina?",
        "resposta_esperada": "Para ser aprovado, o estudante deve ter frequência igual ou superior a 75% do total de horas letivas da disciplina."
    },
    {
        "pergunta": "Qual é a média mínima para aprovação direta nos cursos de graduação?",
        "resposta_esperada": "Nos cursos de graduação, a média parcial deve ser igual ou superior a 7,0 para aprovação direta."
    },
    {
        "pergunta": "Qual é a média mínima para aprovação direta nos cursos técnicos?",
        "resposta_esperada": "Nos cursos técnicos concomitantes e subsequentes, a média parcial deve ser igual ou superior a 6,0 para aprovação direta."
    },
    {
        "pergunta": "Como é calculada a média final para quem faz a avaliação final?",
        "resposta_esperada": "A média final é calculada pela média aritmética entre a média parcial e a nota da avaliação final. O estudante é aprovado se a média final for igual ou superior a 5,0."
    },
    {
        "pergunta": "O que é trancamento de matrícula?",
        "resposta_esperada": "O trancamento de matrícula é o ato formal pelo qual o estudante opta pela interrupção temporária dos estudos, sem perder o vínculo com o IFCE."
    },
    {
        "pergunta": "Por quanto tempo o estudante pode manter a matrícula trancada em cursos semestrais?",
        "resposta_esperada": "O trancamento pode durar até 4 períodos letivos para cursos semestrais."
    },
    {
        "pergunta": "É permitido trancar a matrícula no primeiro período letivo?",
        "resposta_esperada": "Em regra, não é permitido o trancamento no primeiro período letivo. Exceções são feitas apenas para casos comprovados de saúde, serviço militar, transferência compulsória de trabalho, gravidez de risco ou trabalho formal."
    },
    {
        "pergunta": "Qual é o prazo para renovação de matrícula?",
        "resposta_esperada": "A renovação de matrícula deve ser feita on-line pelo Q-Acadêmico dentro dos prazos do calendário acadêmico. Se o estudante perder o prazo, tem até 5 dias letivos para regularizar na CCA."
    },
    {
        "pergunta": "O que acontece se o estudante não renovar a matrícula no prazo?",
        "resposta_esperada": "Se o estudante não renovar a matrícula no prazo on-line e não regularizar em até 5 dias letivos na CCA, sua matrícula será configurada como ABANDONO."
    },
    {
        "pergunta": "O que é progressão parcial?",
        "resposta_esperada": "A Progressão Parcial de Estudos, prevista para os cursos técnicos integrados, é a possibilidade de o estudante ser promovido para o período letivo seguinte mesmo sem rendimento satisfatório em até 2 componentes curriculares. Se reprovar em 3 ou mais, não poderá se matricular no período seguinte."
    },
    {
        "pergunta": "Qual o prazo para solicitar segunda chamada de avaliação?",
        "resposta_esperada": "A segunda chamada deve ser requerida em até 5 dias letivos após a data da avaliação original, com apresentação de documento comprobatório."
    },
    {
        "pergunta": "Qual o prazo para justificar faltas com atestado médico?",
        "resposta_esperada": "O estudante deve protocolar um requerimento em até 5 dias letivos após o primeiro dia de ausência, anexando o atestado médico ou outro documento comprobatório."
    },
    {
        "pergunta": "O que é o Regime de Exercícios Domiciliares?",
        "resposta_esperada": "O Regime de Exercícios Domiciliares permite substituir aulas presenciais por atividades em casa em casos de doença infectocontagiosa, incapacidade física temporária de pelo menos 15 dias ou gravidez, mediante laudo médico com CID."
    },
    {
        "pergunta": "O IFCE reconhece o uso do nome social?",
        "resposta_esperada": "Sim, o IFCE garante o uso do nome social em registros acadêmicos, avaliações e diplomas, mediante solicitação ao gestor de ensino do campus."
    },
    {
        "pergunta": "Como acessar o SEI pela primeira vez?",
        "resposta_esperada": "Para acessar o SEI pela primeira vez, o estudante deve acessar https://ifce.edu.br/sei, clicar em Acesso para Usuário Externo, depois em Esqueci minha senha e informar o e-mail cadastrado no Q-Acadêmico."
    },
    {
        "pergunta": "Qual o formato e tamanho máximo de documentos aceitos no SEI?",
        "resposta_esperada": "Os documentos enviados pelo SEI devem estar em formato PDF com tamanho máximo de 50MB por arquivo."
    },
    {
        "pergunta": "Como o estudante pode acompanhar o andamento de um processo no SEI?",
        "resposta_esperada": "O estudante pode acompanhar seus processos logado no sistema em Recibos Eletrônicos de Protocolo, ou sem login pela Pesquisa Pública em https://ifce.edu.br/sei com o número do processo."
    },
    {
        "pergunta": "É possível trancar apenas uma disciplina?",
        "resposta_esperada": "Sim, nos cursos técnicos (exceto integrados) e de graduação, desde que o estudante permaneça matriculado em pelo menos 12 créditos. Deve ser solicitado nos primeiros 30 dias do período letivo."
    },
    {
        "pergunta": "O que é necessário para solicitar aproveitamento de disciplinas cursadas em outra instituição?",
        "resposta_esperada": "O estudante deve solicitar o aproveitamento na coordenadoria, apresentando histórico escolar e programas das disciplinas. Para ingressantes, o prazo é de até 10 dias letivos após a matrícula."
    },
    {
        "pergunta": "O IFCE permite reingresso de estudante que abandonou o curso?",
        "resposta_esperada": "Sim, o IFCE permite reingresso em oportunidade única, desde que o abandono tenha ocorrido há no máximo 5 anos e existam vagas no curso."
    },
    {
        "pergunta": "Como solicitar mudança de turno?",
        "resposta_esperada": "A mudança de turno pode ser solicitada à coordenadoria com justificativa de trabalho formal ou estágio, dependendo da existência de vagas no turno desejado."
    },
    {
        "pergunta": "Por quanto tempo pode durar o trancamento em cursos anuais?",
        "resposta_esperada": "O trancamento pode durar até 2 períodos letivos para cursos anuais."
    },
    {
        "pergunta": "Como abrir um processo no SEI?",
        "resposta_esperada": "O estudante deve acessar o SEI, ir em Peticionamento, Processo Novo, escolher Aluno: Requerimento Geral, preencher o formulário, anexar os documentos em PDF e clicar em Peticionar."
    },
    {
        "pergunta": "Quantas disciplinas o estudante pode dever para avançar de período pela progressão parcial?",
        "resposta_esperada": "Nos cursos técnicos integrados, o estudante pode avançar devendo até 2 componentes curriculares. Se reprovar em 3 ou mais, não poderá se matricular no período seguinte."
    },
    {
        "pergunta": "Quais documentos podem ser usados para justificar faltas além do atestado médico?",
        "resposta_esperada": "Além do atestado médico, são aceitos documentos comprobatórios de serviço militar e óbito de parente próximo, entre outros previstos no ROD."
    },
    {
    
        "pergunta": "Posso justificar minhas faltas no final do semestre?",
        "resposta_esperada": "Não. A justificativa de faltas deve ser protocolada em até 5 dias letivos após o primeiro dia de ausência. Não é permitido acumular justificativas para o final do semestre."
    },
    {
        "pergunta": "Certificados online contam para solicitar aproveitamento de disciplinas?",
        "resposta_esperada": "Não encontrei essa informação específica no ROD. Para dúvidas sobre documentos aceitos para aproveitamento, consulte a coordenadoria do seu curso."
    },
    {
        "pergunta": "A experiência profissional conta para solicitar aproveitamento de disciplinas?",
        "resposta_esperada": "Não encontrei essa informação específica no ROD. Para dúvidas sobre critérios de aproveitamento, consulte a coordenadoria do seu curso."
    },
    {
        "pergunta": "Se eu reprovar em uma disciplina, posso solicitar aproveitamento novamente?",
        "resposta_esperada": "Não encontrei essa informação específica no ROD. Para dúvidas sobre aproveitamento após reprovação, consulte a coordenadoria do seu curso."
    }
]

# Perguntas cuja informação não existe nos documentos: o comportamento correto
# é o bot se abster. São avaliadas à parte, pois o RAGAS penaliza abstenções
# (answer_relevancy dá 0 para respostas "evasivas" e context_recall não se aplica).
def eh_sem_resposta(q):
    return q["resposta_esperada"].startswith("Não encontrei")

def bot_se_absteve(resposta):
    return "não encontrei" in resposta.lower()

# ============================================================
# COLETA DE RESPOSTAS E CONTEXTOS DO SISTEMA
# ============================================================

total = len(questoes)
print(f"Coletando respostas do RodBot para as {total} questões...")
print("=" * 60)

amostras = []
falhas = []

for i, q in enumerate(questoes):
    print(f"[{i+1}/{total}] {q['pergunta'][:60]}...")

    # Replica responder_rag() sem o try/except, para que erros de API não
    # sejam contabilizados como resposta do bot, e usando os mesmos documentos
    # na geração e na avaliação (uma única chamada ao retriever).
    try:
        docs = retriever.invoke(q["pergunta"])
        contexto = [
            f"[Fonte: {doc.metadata.get('fonte', 'desconhecida')}]\n{doc.page_content}"
            for doc in docs
        ]
        resposta = llm.invoke(
            prompt.format(context="\n\n".join(contexto), question=q["pergunta"])
        ).content.strip()
    except Exception as e:
        print(f"   ERRO: {e}")
        falhas.append({"id": i + 1, "pergunta": q["pergunta"], "erro": str(e)})
        continue

    amostras.append({
        "id": i + 1,
        "pergunta": q["pergunta"],
        "resposta_gerada": resposta,
        "resposta_esperada": q["resposta_esperada"],
        "contextos": contexto,
        "sem_resposta": eh_sem_resposta(q),
    })

print(f"\nColeta concluída! {len(amostras)} respostas, {len(falhas)} falhas.")
print("=" * 60)

# ============================================================
# AVALIAÇÃO RAGAS (apenas perguntas com resposta nos documentos)
# ============================================================

respondiveis = [a for a in amostras if not a["sem_resposta"]]
sem_resposta = [a for a in amostras if a["sem_resposta"]]

print(f"\nIniciando avaliação RAGAS ({len(respondiveis)} questões)...")

dataset = Dataset.from_dict({
    "question": [a["pergunta"] for a in respondiveis],
    "answer": [a["resposta_gerada"] for a in respondiveis],
    "contexts": [a["contextos"] for a in respondiveis],
    "ground_truth": [a["resposta_esperada"] for a in respondiveis],
})

METRICAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

resultado = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=llm_factory(MODELO_JUIZ, client=OpenAI()),
    # Passadas explicitamente: no ragas 0.4.3 as embeddings criadas
    # automaticamente não têm embed_query() e o answer_relevancy dá NaN
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
)

# Notas por pergunta; o RAGAS devolve NaN quando o juiz falha em uma amostra
df = resultado.to_pandas()
for a, (_, linha) in zip(respondiveis, df.iterrows()):
    a["notas"] = {
        m: (None if math.isnan(linha[m]) else float(linha[m])) for m in METRICAS
    }

medias = {}
for m in METRICAS:
    valores = df[m].dropna()
    medias[m] = {
        "media": float(valores.mean()) if len(valores) else None,
        "amostras_validas": int(len(valores)),
    }

# Abstenção correta nas perguntas sem resposta nos documentos
for a in sem_resposta:
    a["absteve"] = bot_se_absteve(a["resposta_gerada"])
acertos_abstencao = sum(a["absteve"] for a in sem_resposta)

# ============================================================
# EXIBIÇÃO DOS RESULTADOS
# ============================================================

def fmt(m):
    v = medias[m]
    if v["media"] is None:
        return "   n/a"
    return f"{v['media']:.4f}  (n={v['amostras_validas']})"

print("\n" + "=" * 60)
print(f"RESULTADO DA AVALIAÇÃO RAGAS (juiz: {MODELO_JUIZ})")
print("=" * 60)
print(f"Faithfulness (Fidelidade):       {fmt('faithfulness')}")
print(f"Answer Relevancy (Relevância):   {fmt('answer_relevancy')}")
print(f"Context Precision (Precisão):    {fmt('context_precision')}")
print(f"Context Recall (Cobertura):      {fmt('context_recall')}")
print("-" * 60)
print(f"Abstenção correta (sem resposta): {acertos_abstencao}/{len(sem_resposta)}")
if falhas:
    print(f"Falhas na coleta (excluídas):     {len(falhas)}")
print("=" * 60)

# Salva os resultados em JSON
resultados_json = {
    "modelo_juiz": MODELO_JUIZ,
    "metricas": medias,
    "abstencao": {
        "acertos": acertos_abstencao,
        "total": len(sem_resposta),
    },
    "questoes": amostras,
    "falhas": falhas,
}

with open("resultados_ragas.json", "w", encoding="utf-8") as f:
    json.dump(resultados_json, f, ensure_ascii=False, indent=2)

print("\nResultados salvos em: resultados_ragas.json")

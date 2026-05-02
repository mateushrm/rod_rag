import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carregar PDFs
pdfs = [
    "ROD_2025.pdf",
    "Manual_de_Peticionamento_Eletronico_de_Processos_Aluno.pdf",
    "cartilha_usuario_sei.pdf"
]

documents = []
for pdf in pdfs:
    path = os.path.join(BASE_DIR, pdf)
    if not os.path.exists(path):
        print(f"AVISO: arquivo não encontrado: {pdf}")
        continue
    loader = PyPDFLoader(path)
    docs = loader.load()

    for doc in docs:
        # Limpar texto
        texto = doc.page_content.replace("\n", " ")
        texto = " ".join(texto.split())
        doc.page_content = texto

        # Adiciona a fonte nos metadados
        doc.metadata["fonte"] = pdf

    documents.extend(docs)
    print(f"✔ Carregado: {pdf} ({len(docs)} páginas)")

print(f"\nTotal de páginas carregadas: {len(documents)}")

# Dividir em chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

texts = text_splitter.split_documents(documents)
print(f"Total de chunks gerados: {len(texts)}")

# Embeddings locais
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Criar banco vetorial
persist_directory = os.path.join(BASE_DIR, "db_rod")

vectorstore = Chroma.from_documents(
    texts,
    embeddings,
    persist_directory=persist_directory
)

print("\nIndexação concluída com sucesso!")
print(f"Banco vetorial salvo em: {persist_directory}")
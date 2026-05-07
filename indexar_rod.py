import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "pdfs")

# Carregar pdfs
documents = []
for arquivo in os.listdir(PDF_DIR):
    if arquivo.endswith(".pdf"):
        path = os.path.join(PDF_DIR, arquivo)
        loader = PyPDFLoader(path)
        docs = loader.load()

        for doc in docs:
            texto = doc.page_content.replace("\n", " ")
            texto = " ".join(texto.split())
            doc.page_content = texto
            doc.metadata["fonte"] = arquivo

        documents.extend(docs)
        print(f"Carregado: {arquivo} ({len(docs)} páginas)")

print(f"\nTotal de páginas: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
texts = text_splitter.split_documents(documents)
print(f"Total de chunks: {len(texts)}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectorstore = Chroma.from_documents(
    texts,
    embeddings,
    persist_directory=os.path.join(BASE_DIR, "db_rod")
)

print("\nIndexação concluída!")
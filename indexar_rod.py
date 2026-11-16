import os
import re
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
DB_DIR = os.path.join(BASE_DIR, "db_rod")

# Carregar pdfs
documents = []
for arquivo in os.listdir(PDF_DIR):
    if arquivo.endswith(".pdf"):
        path = os.path.join(PDF_DIR, arquivo)
        loader = PyPDFLoader(path)
        docs = loader.load()

        for doc in docs:
            texto = re.sub(r"[ \t]+", " ", doc.page_content)
            texto = re.sub(r"\n{3,}", "\n\n", texto)
            doc.page_content = texto.strip()
            doc.metadata["fonte"] = arquivo
        
        documents.extend(docs)
        print(f"Carregado: {arquivo} ({len(docs)} páginas)")

print(f"\nTotal de páginas: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=200,
     separators=["\nArt. ", "\n\n", "\n", ". ", " ", ""],
)
texts = text_splitter.split_documents(documents)
print(f"Total de chunks: {len(texts)}")

for t in texts:
    t.page_content = f"[{t.metadata['fonte']} - pág. {t.metadata.get('page', 0) + 1}] {t.page_content}"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma.from_documents(
    texts,
    embeddings,
    persist_directory=os.path.join(BASE_DIR, "db_rod")
)

print("\nIndexação concluída!")
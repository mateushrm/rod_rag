# Imagem base Python
FROM python:3.11-slim

# Diretório de trabalho
WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala torch CPU primeiro (índice separado)
RUN pip install --no-cache-dir \
    torch==2.10.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copia e instala demais dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia os arquivos do projeto
COPY api.py .
COPY telegram_bot.py .
COPY consultar_rod.py .
COPY faq.json .
COPY ROD_2025.pdf .
COPY cartilha_usuario_sei.pdf .
COPY Manual_de_Peticionamento_Eletronico_de_Processos_Aluno.pdf .
COPY db_rod/ ./db_rod/

# Expõe a porta da API
EXPOSE 8000

# Imagem base Python
FROM python:3.11-slim

# Diretório de trabalho
WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala torch CPU (índice separado)
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
COPY pdfs/ ./pdfs/
COPY db_rod/ ./db_rod/

# Cria o script de inicialização diretamente no container (sem problema de encoding)
RUN printf '#!/bin/bash\nuvicorn api:app --host 0.0.0.0 --port 8000 &\npython telegram_bot.py\n' > /app/start.sh \
    && chmod +x /app/start.sh

# Expõe a porta da API
EXPOSE 8000

CMD ["/app/start.sh"]
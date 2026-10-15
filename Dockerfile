
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

# Install CPU-only PyTorch first to avoid large CUDA dependencies
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.7.1"

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py backend.py ./
COPY peice.pdf ./

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

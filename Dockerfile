FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home --uid 10001 aura && mkdir -p /app/data && chown -R aura:aura /app
USER aura
EXPOSE 8000 8501
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

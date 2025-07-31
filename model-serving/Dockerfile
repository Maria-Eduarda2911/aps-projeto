FROM python:3.10

WORKDIR /app

RUN pip install mlflow scikit-learn pandas gunicorn

# Copia o modelo treinado
COPY mlruns/1 ./model

EXPOSE 5000

CMD ["mlflow", "models", "serve", "-m", "model", "-h", "0.0.0.0", "-p", "5000"]
from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
import os

app = Flask(__name__)

MODEL_SERVICE = os.getenv("MODEL_SERVICE", "http://model:5001/invocations")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400
    
    file = request.files['file']
    try:
        df = pd.read_csv(file)
        if "obesity_rate" in df.columns:
            df = df.drop(columns=["obesity_rate"])
    except Exception as e:
        return jsonify({"error": f"Erro ao ler arquivo: {str(e)}"}), 400
    
    try:
        data = {
            "dataframe_split": {
                "columns": df.columns.tolist(),
                "data": df.values.tolist()
            }
        }
        response = requests.post(MODEL_SERVICE, json=data)
        if response.status_code != 200:
            return jsonify({"error": f"Erro no modelo: {response.text}"}), 500
        
        predictions = response.json().get("predictions", [])
        df["prediction"] = predictions
        return jsonify({"predictions": df.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": f"Erro ao processar: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
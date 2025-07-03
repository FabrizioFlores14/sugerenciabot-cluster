from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.cluster import KMeans
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/sugerir', methods=['POST'])
def sugerir():
    data = request.get_json()

    if not data or 'productos' not in data:
        return jsonify({'error': 'Formato incorrecto. Se espera {"productos": [...] }'}), 400

    df = pd.DataFrame(data['productos'])

    if df.empty or not all(col in df.columns for col in ['producto', 'stock', 'riesgo', 'repeticiones']):
        return jsonify({'error': 'Faltan columnas necesarias: producto, stock, riesgo, repeticiones'}), 400

    if len(df) < 3:
        return jsonify({'error': 'Se requieren al menos 3 productos para aplicar clustering.'}), 400

    # Normalizar datos numéricos
    df['riesgo'] = df['riesgo'].astype(float)
    df['stock'] = df['stock'].astype(float)
    df['repeticiones'] = df['repeticiones'].astype(int)

    # Solo columnas para clustering
    X = df[['riesgo', 'stock', 'repeticiones']]

    # K-means con 3 clústers
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    # Identificar clúster más crítico: alto riesgo, bajo stock, alta repetición
    centroides = pd.DataFrame(kmeans.cluster_centers_, columns=['riesgo', 'stock', 'repeticiones'])
    centroides['cluster'] = centroides.index

    # Ranking ponderado del centroide
    centroides['critico'] = (
        (centroides['riesgo'] * 0.4) +
        (1 / (centroides['stock'] + 1) * 0.4) +
        (centroides['repeticiones'] * 0.2)
    )

    cluster_critico = centroides.sort_values('critico', ascending=False).iloc[0]['cluster']
    sugerencias = df[df['cluster'] == cluster_critico][['producto', 'riesgo', 'stock', 'repeticiones']]

    return jsonify({'sugerencias': sugerencias.to_dict(orient='records')})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


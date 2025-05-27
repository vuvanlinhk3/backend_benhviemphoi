from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from models import init_db
from db import insert_analysis, get_all_analyses
from utils import analyze_image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Gọi init_db() khi app khởi tạo
init_db()

@app.route('/uploads/<filename>')
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    result, pneumonia_prob, normal_prob, analyzed_at = analyze_image(image_path)

    heatmap_path = ''  # Nếu có heatmap thì cập nhật ở đây
    insert_analysis(image_path, heatmap_path, result, pneumonia_prob, normal_prob, analyzed_at)

    return jsonify({
        "result": result,
        "pneumonia_prob": pneumonia_prob,
        "normal_prob": normal_prob,
        "analyzed_at": analyzed_at
    })

@app.route('/history', methods=['GET'])
def history():
    rows = get_all_analyses()
    analyses = []
    for row in rows:
        analyses.append({
            "id": row[0],
            "image_path": row[1].replace('\\', '/'),
            "heatmap_path": row[2].replace('\\', '/'),
            "result": row[3],
            "pneumonia_prob": row[4],
            "normal_prob": row[5],
            "analyzed_at": row[6]
        })
    return jsonify(analyses)

if __name__ == '__main__':
    app.run(debug=True)

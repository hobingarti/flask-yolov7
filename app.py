import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Set a directory to save the uploaded images
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def home():
    return "Flask API enabled, upload your images to /upload"


@app.route('/upload', methods=['POST', 'GET'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return jsonify({'message': f'File {file.filename} uploaded successfully', 'path': filepath}), 200

    return jsonify({'error': 'File not saved'}), 500


if __name__ == "__main__":
    # Bind to 0.0.0.0 to make the app accessible externally
    app.run(host="0.0.0.0", port=5000)

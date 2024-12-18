import os
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

# Set a directory to save the uploaded images
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def home():
    return "Flask API enabled, upload your images to /upload"

@app.route("/detect", methods=['GET'])
def detect():
    # os.system("python detect.py --weights .\weight\pillar.pt --conf 0.2 --source .\uploads\pilar-16-_JPG.rf.ccf46654b9e0c5d20e3d1b50ae9c1847.jpg --project runs/pilar")
    os.system("python detect.py  --weights ./weight/pillar.pt --conf 0.2  --source ./uploads/pilar-4-_JPG.rf.27c2c37bf9eab1626fa8e943c325b0ee.jpg --project runs/pilar --name result --exist-ok")
    return "Processing is done"

@app.route("/send-value", methods=['POST'])
def send_value():
    content = request.form['imageData']
    return content

@app.route("/send-image", methods=['POST'])
def send_image():
    content = request.form['imageData']
    fileName = request.form['imageName']
    
    # content = content.replace('\n', '').replace('\r', '')
    image_data = base64.b64decode(content)
    with open('uploads/'+fileName, "wb") as f:
        f.write(image_data)
    
    # g = open(fileName, "w")
    # # g.write(content)
    # g.write(base64.b64decode(content))
    # g.close()
    
    return "Image saved as "+fileName


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

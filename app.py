from flask import Flask, render_template, request, jsonify, redirect, url_for
from .cv2 import *
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import os
from flask import flash

app = Flask(__name__)

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def process_image(image_path):
    image = cv2.imread(image_path)
    boxes, _ = mtcnn.detect(image)

    if boxes is not None and boxes[0] is not None:
        box = boxes[0]
        face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        face_tensor = transform(face).unsqueeze(0).to(device)
        embedding_image = resnet(face_tensor).detach().cpu().numpy()[0]
        return embedding_image
    else:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare_faces():
    image_file = request.files['image']
    video_file = request.files['video']

    print("Image File Received:", image_file)
    print("Video File Received:", video_file)


    image_path = 'temp_image.jpg'
    video_path = 'temp_video.mp4'

    image_file.save(image_path)
    video_file.save(video_path)

    embedding_image = process_image(image_path)

    if embedding_image is not None:
        cap = cv2.VideoCapture(video_path)
        match_found = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            boxes, _ = mtcnn.detect(frame)

            if boxes is not None and boxes[0] is not None:
                box = boxes[0]
                face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((160, 160)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
                face_tensor = transform(face).unsqueeze(0).to(device)
                embedding_video = resnet(face_tensor).detach().cpu().numpy()[0]

                similarity = np.dot(embedding_image, embedding_video) / (np.linalg.norm(embedding_image) * np.linalg.norm(embedding_video))
                similarity_percentage = round(similarity * 100, 2)

                if similarity_percentage >= 70:
                    match_found = True
                    break

        cap.release()

        os.remove(image_path)
        os.remove(video_path)

        if match_found:
            return jsonify({'message': 'Matching: Yes'})
        else:
            return redirect(url_for('details'))
    else:
        return jsonify({'message': 'No faces detected in the image.'})

@app.route('/details', methods=['GET', 'POST'])
def details():
    if request.method == 'POST':
        # Get user input from the form
        name = request.form['name']
        address = request.form['address']
        email = request.form['email']
        phone = request.form['phone']

        # Process the data or save it to a database
        # Here we are just printing the data for demonstration
        print(f"Name: {name}, Address: {address}, Email: {email}, Phone: {phone}")
        
        flash('details provided successfully!', 'success')
        return redirect(url_for('index'))


    return render_template('details.html')

if __name__ == '__main__':
    app.run(debug=True)

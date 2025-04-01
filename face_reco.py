from keras.models import load_model
from keras_facenet import FaceNet
import cv2
import numpy as np
from numpy import linalg as LA
import tensorflow as tf
from facenet_pytorch import MTCNN


def preprocess_image(image_path):
    # Load the image using OpenCV
    print(image_path)
    img = cv2.imread(image_path)

    # Convert the image to RGB (FaceNet expects RGB images)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to 160x160 pixels
    img = cv2.resize(img, (224, 224))

    # Normalize the pixel values
    # img = img.astype('float32') / 255.0

    # Expand dimensions to match the input shape of FaceNet (1, 160, 160, 3)
    img = np.expand_dims(img, axis=0)

    return img


def get_face_embedding(model, image_path):
    # Preprocess the image
    img = preprocess_image(image_path)

    # Generate the embedding
    embedding = model.predict(img)

    return embedding


def compare_faces(embedding1, embedding2, threshold=0.15):
    # Compute the Euclidean distance between the embeddings
    distance = LA.norm(embedding1 - embedding2)

    # Compare the distance to the threshold
    if distance < threshold:
        print("Face Matched.")
    else:
        print("Faces are different.")

    return distance


def compare(img1, img2):
    # Load the pre-trained FaceNet model
    model = tf.keras.applications.ResNet50(weights='imagenet')
#    model = FaceNet()
    print("Model Loaded Successfully")

    # Get embeddings for two images
    embedding1 = get_face_embedding(model, img1)
    embedding2 = get_face_embedding(model, img2)

    # Compare the two faces
    distance = compare_faces(embedding1, embedding2)

    print(f"Euclidean Distance: {distance}")

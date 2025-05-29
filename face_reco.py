from keras.models import load_model
from keras_facenet import FaceNet
import cv2
import numpy as np
from numpy import linalg as LA
import tensorflow as tf
from facenet_pytorch import MTCNN
import heatmap_2
import os


class face_reco:
    def __init__(self):
        self.heatmap = heatmap_2.GradCAM()

    def train(self, path):
        self.heatmap.train(path)

    def preprocess_image(self, image_path):
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

    def write_to_path(self, image, path):
        num_files = len(os.listdir(path))
        filename = path + "heatmap" + ".jpg"
        print("writing " + filename)
        cv2.resize(image, (200, 200))
        cv2.imwrite(filename, image)

    def get_face_embedding(self, image_path):
        # Generate the embedding
        (embedding, heatmap) = self.heatmap.prediction(image_path)

        self.write_to_path(
            heatmap, "/Users/clem/Projets/prog/gif5000/")
        return embedding

    def compare_faces(self, embedding1, embedding2, threshold=0.15):
        # Compute the Euclidean distance between the embeddings
        distance = LA.norm(embedding1.detach().numpy() -
                           embedding2.detach().numpy())

        # Compare the distance to the threshold
        if distance < threshold:
            print("Face Matched.")
        else:
            print("Faces are different.")

        return distance

    def compare(self, img1, img2):
        # Get embeddings for two images
        embedding1 = self.get_face_embedding(img1)
        embedding2 = self.get_face_embedding(img2)

        # Compare the two faces
        self.distance = compare_faces(embedding1, embedding2)

        print(f"Euclidean Distance: {distance}")

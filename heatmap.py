from keras.models import load_model
import tensorflow as tf
import face_reco
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import ssl
from PIL import Image

# Goal of this file is to generate a heatmap revealing the vectors of FaceNet
# According to https://arxiv.org/pdf/2109.06467 we want to use the backward gradient
# of the triplet loss function.
# We will use FaceNet pretrained model.
# Triplet loss function takes face encoding of 3 pictures: anchor, negative and positive.
# anchor and positive are the same whereas anchor and negative are differents persons
DATASET = "/Users/clem/Projets/prog/gif5000/set/"
ANCHOR = "/Users/clem/Projets/prog/gif5000/set/anchor/"
POSITIVE = "/Users/clem/Projets/prog/gif5000/set/positive/"
NEGATIVE = "/Users/clem/Projets/prog/gif5000/set/negative/"


def get_triplet_embedding(triplet):
    triplet_embedding = []

    triplet_embedding = (get_image_embedding(model, triplet[0], mtcnn),
                         get_image_embedding(model, triplet[1], mtcnn),
                         get_image_embedding(model, triplet[2], mtcnn))
    print(triplet_embedding)


def get_image_embedding(model, image_path, cropper):
    # Preprocess the image
    img = Image.open(image_path)
    img_cropped = cropper(img)

    # Generate the embedding
    embedding = model(img_cropped.unsqueeze(0))

    return embedding


def get_from_dataset(subdir):
    path = DATASET + subdir
    # TODO: Change to early return when not(anchor, positive, negative)
    if subdir == "anchor":
        files = os.listdir(path)
        return files
    elif subdir == "positive":
        files = os.listdir(path)
        return files
    elif subdir == "negative":
        files = os.listdir(path)
        return files
    else:
        return None


def get_triplets():
    triplets = []
    number_pos_neg = min(len(positive_image_list), len(negative_image_list))
    num_anchor = len(os.listdir(DATASET + "anchor/"))
    print(number_pos_neg)
    for i in range(0, number_pos_neg):
        if (i != 0):
            anchor_num = num_anchor % i
        else:
            anchor_num = 0

        triplets.append((ANCHOR + os.listdir(ANCHOR)[anchor_num], POSITIVE + os.listdir(
            POSITIVE)[i], NEGATIVE + os.listdir(NEGATIVE)[i]))
        print(triplets)
        return triplets


positive_preco_image_list = []
negative_preco_image_list = []
anchor_list = []

anchor_list = get_from_dataset("anchor")
positive_image_list = get_from_dataset("positive")
negative_image_list = get_from_dataset("negative")

ssl._create_default_https_context = ssl._create_unverified_context
# Use MCTNN to crop images for now. Maybe later to detect faces.
mtcnn = MTCNN(image_size=160, margin=0)
# model = tf.keras.applications.ResNet50(weights='imagenet')
model = InceptionResnetV1(pretrained='vggface2').eval()

# Naive Triplet preparation
anchor_image_list = []
for image in anchor_image_list:
    anchor_list.append(get_image_embedding(
        model, DATASET + "anchor/" + image, mtcnn))

positive_list = []
for image in positive_image_list:
    positive_list.append(get_image_embedding(
        model, DATASET + "positive/" + image, mtcnn))

negative_list = []
for image in negative_preco_image_list:
    negative_list.append(face_reco.get_image_embedding(
        model, DATASET + "negative/" + image, mtcnn))

triplets_list = get_triplets()
for triplet in triplets_list:
    embeddings = get_triplet_embedding(triplet)

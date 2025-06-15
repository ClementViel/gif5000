import json
import urllib.request
import ssl
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pyxtend import struct
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.resnet import ResNet50_Weights

import vgg_super
IMAGENET_MEAN_VALUES = [0.485, 0.456, 0.406]
IMAGENET_STD_VALUES = [0.229, 0.224, 0.225]
IMAGENET_CLASSES_URL = "https://raw.githubusercontent.com/jss367/files/main/imagenet_classes.json"


class GradCAM:
    def __init__(self):
        self.gradients = None
        self.activations = None
#        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.model = vgg_super.VGG()
#        layer_name, target_layer = self.find_last_conv_layer(self.model)

        # Register hooks for gradients and activations
#        target_layer.register_forward_hook(self.forward_hook)
#        target_layer.register_full_backward_hook(self.full_backward_hook)

#    def forward_hook(self, module, input, output):
#        self.activations = output.detach()
#
#    def full_backward_hook(self, module, grad_input, grad_output):
#        self.gradients = grad_output[0].detach()
#

    def compute_heatmap(self, input_batch, class_idx=None):
        # Forward pass
        self.model.eval()
        logits = self.model(input_batch)
#        self.model.zero_grad()
        logits[:, 1].backward()
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
            #            class_idx = logits.argmax(logits, dim=1).item()

        gradients = self.model.get_activation_gradient()
        print(gradients)
        activations = self.model.get_activation(input_batch).detach()
        # Compute gradients for the target class
        one_hot_output = torch.zeros_like(logits)
        one_hot_output[0, class_idx] = 1
 #       logits.backward(gradient=one_hot_output)

        # Compute Grad-CAM heatmap resnet
       # weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        # heatmap = torch.sum(weights * activations, dim=1, keepdim=True)
        # heatmap = torch.sum(weights * activations, dim=1)

        weights = torch.mean(gradients, dim=[0, 2, 3])
        # weight the channels by corresponding gradients
        for i in range(len(weights)):
            activations[:, i, :, :] *= weights[i]
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)  # ReLU removes negative values
        heatmap /= torch.max(heatmap)  # Normalize to [0, 1]

        # Get the predicted class probability
        probs = torch.softmax(logits, dim=1)
        predicted_prob = probs[0, class_idx].item()

        return heatmap.squeeze().cpu().numpy(), class_idx, predicted_prob

    def find_last_conv_layer(self, model: nn.Module) -> tuple:
        last_conv_layer_name = None
        last_conv_layer = None
        for layer_name, layer in model.named_modules():
            print("layer" + layer_name)
            if isinstance(layer, nn.Conv2d):
                print("found layer " + layer_name)
                last_conv_layer_name = layer_name
                last_conv_layer = layer

        return last_conv_layer_name, last_conv_layer

    def generate_heatmap(self, img_path: str, heatmap: np.ndarray) -> None:
        # Read the image from the given file path
        img = cv2.imread(img_path)

        # Resize the heatmap to match the size of the original image
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # Normalize the heatmap values to the range [0, 255] and cast to uint8
        heatmap = np.uint8(255 * heatmap)

        # Apply the JET colormap to the heatmap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Blend the original image with the heatmap (60% original, 40% heatmap)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        return superimposed_img
        # Display the blended image in RGB format
      #  plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
       # plt.axis('off')
        # plt.show()

    # img_path = "/Users/clem/Projets/prog/gif5000/photo0.jpg"

    def train(self, path):
        self.model.load_dataset(path)
        self.model.train_model()

    def save_model(self):
        torch.save(self.model.state_dict(),
                   "/Users/clem/Projets/prog/gif5000/model")

    def load_model(self):
        self.model.load_state_dict(torch.load(
            "/Users/clem/Projets/prog/gif5000/model", weights_only=True))
        print("model loaded")

    def prediction(self, img_path):
        input_image = Image.open(img_path)
        ssl._create_default_https_context = ssl._create_unverified_context
        self.model.eval()

        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN_VALUES,
                                     std=IMAGENET_STD_VALUES),
            ]
        )

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        logits = self.model(input_batch)
        probs = torch.softmax(logits, dim=1)
        pred_class_idx = torch.argmax(probs, dim=1).item()
        predicted_prob = probs[0, pred_class_idx].item()
        class_labels = ["anchor", "negative"]
#        class_labels = json.loads(requests.get(IMAGENET_CLASSES_URL).text)
        struct(class_labels, examples=True)

        predicted_class_name = class_labels[pred_class_idx]

        heatmap, predicted_class_idx, predicted_prob = self.compute_heatmap(
            input_batch)
        heatmap_image = self.generate_heatmap(img_path, heatmap)
        print("Predicted class is " + predicted_class_name)
        print("Precision : ")
        print(predicted_prob)
        return (logits, heatmap_image)

# import torch
# from torchvision import transforms
# from PIL import Image

# classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# def get_prediction(image_bytes):
#     transform = transforms.Compose([transforms.Resize((64, 64)), 
#                                     transforms.ToTensor(), 
#                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#     device = torch.device('cpu')
#     model = torch.load('asl_99.94.pth', map_location=device)
#     # Load the state_dict into the model
#     model.load_state_dict(state_dict)
#     model.eval()
#     tensor = transform(Image.open(image_bytes)).unsqueeze(0)
#     outputs = model(tensor)
#     _, predicted = torch.max(outputs, 1)
#     return classes[predicted]


# from PIL import Image
# # image = Image.open('./asl-alphabet/asl_alphabet_test/asl_alphabet_test/K_test.jpg')

# image = Image.open('./hande.jpg')

# print(get_prediction(image))


import torch
import torchvision.transforms as tt
import os

from modellib import ResNet9, to_device

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
class ASLModel:
    def __init__(self, classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = classes
        self.model = self.load_model()
    
    def load_model(self):
        model = ResNet9(3, len(self.classes))
        model.load_state_dict(torch.load('asl_99.94.pth', map_location=torch.device('cpu')))
        model.eval()
        return to_device(model, self.device)
    
    def get_prediction(self, tensori):
        # Load image and convert to tensor
        xb = image.unsqueeze(0)
        xb = to_device(xb, self.device)
        
        # Make prediction
        with torch.no_grad():
            preds = F.softmax(self.model(xb), dim=1)
        
        # Get predicted class and confidence score
        max_prob, preds_idx = torch.max(preds, dim=1)
        pred_class = self.classes[preds_idx]
        confidence = max_prob.item()
        print(pred_class)
        return pred_class, confidence

if __name__ == "__main__":
    import cv2
    import numpy as np
    from PIL import Image

    # Load image
    image = Image.open('hande.jpg')


    # Load model
    model = ASLModel(classes)

    # Make prediction
    prediction = model.get_prediction(image)

    print(prediction)

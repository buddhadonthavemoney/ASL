import torch
from torchvision import transforms
from PIL import Image

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

def get_prediction(image_bytes):
    transform = transforms.Compose([transforms.Resize((64, 64)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    device = torch.device('cpu')
    model = torch.load('asl_99.94.pth', map_location=device)
    model.eval()
    tensor = transform(Image.open(image_bytes)).unsqueeze(0)
    outputs = model(tensor)
    _, predicted = torch.max(outputs, 1)
    return classes[predicted]
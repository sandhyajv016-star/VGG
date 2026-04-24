import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# Load model
model = models.vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, 3)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Labels
classes = ["cat", "dog", "rat"]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]

# Test
img_path = "test.jpg"
print("Prediction:", predict(img_path))
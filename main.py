import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import sys
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def predict(image_path):
    model_path = 'model.pth'
    
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일 '{model_path}'이 없습니다. 먼저 main.py를 실행해 모델을 학습시켜주세요.")
        return


    model = SimpleCNN().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Model Error: {e}")
        return
    model.eval()


    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Can't find Image File '{image_path}'")
        return
    
    image_tensor = transform(image).unsqueeze(0).to(device)


    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        classes = ['handsome', 'Not'] 
        result = classes[predicted.item()]
        score = probabilities[0][predicted.item()].item()

    print(f"Image: {image_path}")
    print(f"Result: ** {result} ** (Percent: {score*100:.2f}%)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("")
    else:
        predict(sys.argv[1])
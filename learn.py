import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
num_epochs = 5
batch_size = 64
learning_rate = 0.001


def get_data_loaders(data_dir='./dataset', test_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])


    full_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

    if len(full_dataset.classes) < 2:
        raise ValueError(f"Found less than 2 class folders in '{data_dir}'. Please check if folders like 'handsome' and 'Not' exist.")
    
    test_size = int(len(full_dataset) * test_split)
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset loaded from '{data_dir}'. Classes: {full_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    return train_loader, test_loader

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

def main():
    print("Please ensure 'handsome' and 'Not' folders exist in the same directory as the script.")
    print("Example structure: ./handsome/1.jpg, ./Not/1.jpg\n")
    
    print(f"Using device: {device}")
    
    try:
        train_loader, test_loader = get_data_loaders(data_dir='.')
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\nData loading error: {e}")
        print("Please check the folder structure. Ensure class-specific folders like 'handsome' and 'Not' exist in the current directory and contain image files.")
        return
    
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Training Started...")
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    model_path = 'model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining finished. Model saved to {model_path}")

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'\nAccuracy of the model on the {total} test images: {100 * correct / total:.2f} %')

if __name__ == '__main__':
    main()
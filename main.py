import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data_loader import load_mnist
import os
import cv2
# Attention mechanism
class Attention(nn.Module):
    """
    Attention mechanism
    Formular: 
        attention_weights = sigmoid(conv2d(x))
        x = x * attention_weights
    """
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights, attention_weights

# CNN with Attention
class CNNWithAttention(nn.Module):
    """
    CNN with Attention
    Formular: 
        x = conv2d(x)
        x, attention_weights = attention(x)
        x = x * attention_weights
        x = max_pool2d(x)
        x = conv2d(x)
        x = max_pool2d(x)
        x = attention(x)
        x = x.view(x.size(0), -1)
        x = linear(x)
    """
    def __init__(self):
        super(CNNWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.attention = Attention(64)
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x, attention_weights = self.attention(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, attention_weights

# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Batch: {batch_idx}/{len(train_loader)} Loss: {loss.item():.6f}')

# Testing function
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Print the predicted results
            print(f'Predicted: {pred.squeeze().cpu().numpy()}, Actual: {target.cpu().numpy()}')
            
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

# Visualize attention map
def visualize_attention(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, attention_weights = model(data)
            
            # Plot original image and blended attention map
            plt.figure(figsize=(10, 5))
            original_image = data[0].cpu().squeeze()
            attention_map = attention_weights[0].cpu().squeeze()
            
            plt.subplot(1, 2, 1)
            plt.imshow(original_image, cmap='gray')
            plt.title('Original Image')
            
            plt.subplot(1, 2, 2)
            plt.imshow(original_image, cmap='gray')
            plt.imshow(attention_map, cmap='hot', alpha=0.5)  # Blend attention map
            plt.title('Blended Attention Map')
            plt.colorbar()
            plt.show()
            break

# Main training loop
def main():
    import cv2
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_mnist()
    model = CNNWithAttention().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Load the model if it exists
    model_path = 'mnist_cnn_attention.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        num_epochs = 10
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            train(model, train_loader, optimizer, criterion, device)
            test(model, test_loader, criterion, device)
        
        # Save the model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Create the images directory if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')

    # Visualize attention using OpenCV and save images
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, attention_weights = model(data)
            
            original_image = data[0].cpu().squeeze().numpy()
            attention_map = attention_weights[0].cpu().squeeze().numpy()
            
            # Normalize to 0-255 range for visualization
            original_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            attention_map = cv2.normalize(attention_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Create heatmap
            heatmap = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
            
            # Ensure both images have the same size and number of channels
            if original_image.shape[:2] != heatmap.shape[:2]:
                heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

            if len(original_image.shape) == 2:  # If original_image is grayscale
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

            # Now blend the images
            blended = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

            # Concatenate images horizontally
            concatenated_image = cv2.hconcat([original_image, heatmap, blended])

            # Save the concatenated image
            label = target[0].item()
            image_path = f'images/label_{label}.png'
            cv2.imwrite(image_path, concatenated_image)
            print(f'Saved image to {image_path}')

            # Display the concatenated image
            cv2.imshow('Concatenated Image', concatenated_image)
            cv2.waitKey(0)

if __name__ == "__main__":
    main()

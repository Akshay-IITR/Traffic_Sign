import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# Custom Dataset for Test Data
# -------------------------------
class CSVImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file, sep=';')
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Filename'])
        image = Image.open(img_path).convert("RGB")  # supports .ppm too

        # Crop image according to ROI
        x1, y1, x2, y2 = int(row['Roi.X1']), int(row['Roi.Y1']), int(row['Roi.X2']), int(row['Roi.Y2'])
        image = image.crop((x1, y1, x2, y2))

        label = int(row['ClassId'])
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------------------
# Data Loader Function
# -------------------------------
def load_data(batch_size):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # Paths
    train_dir = "Training"
    test_csv = "GT-final_test.csv"
    test_img_dir = "Images"

    # Training images are inside class folders (use ImageFolder)
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = CSVImageDataset(test_csv, test_img_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# -------------------------------
# Train Model
# -------------------------------
def train_model(model, train_loader, test_loader, device, epochs, lr, save_dir, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{model_name}_best.pth")

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple) or hasattr(outputs, 'logits'):
                outputs = outputs.logits  # extract main output
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | {model_name} | Loss: {running_loss/total:.4f} | Train Acc: {train_acc:.4f}")

        # Evaluate
        acc = evaluate(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"{model_name}: New best model saved with accuracy {acc:.4f}")

        ### Saving Accuracy Round-wise
        accuracy_log = {"Epoch": epoch+1, "Train_Accuracy": f"{train_acc:.4f}", "Test_Accuracy": f"{acc:.4f}"}
        with open(f"results/Roundwise_Accuracy_GTSRB_{model_name}.json", "a") as f:
            json.dump(accuracy_log, f, indent=4)

    return checkpoint_path, best_acc

# -------------------------------
# Evaluate
# -------------------------------
def evaluate(model, dataloader, device):
    model.eval()
    preds, labels_list = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple) or hasattr(outputs, 'logits'):
                outputs = outputs.logits  # extract main output
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    acc = accuracy_score(labels_list, preds)
    return acc

# -------------------------------
# Metrics
# -------------------------------
def calculate_metrics(model, dataloader, device, class_names, save_dir, model_name):
    model.eval()
    preds, labels_list = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple) or hasattr(outputs, 'logits'):
                outputs = outputs.logits  # extract main output
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_list, preds)
    precision = precision_score(labels_list, preds, average='weighted', zero_division=0)
    recall = recall_score(labels_list, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels_list, preds, average='weighted', zero_division=0)
    report = classification_report(labels_list, preds, target_names=class_names, output_dict=True)

    metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "precision": precision, "recall": recall, "f1_score": f1, "classification_report": report}, f, indent=4)

    return acc, precision, recall, f1

# -------------------------------
# Main Benchmark
# -------------------------------
def main():
    args = get_args()
    print(f"Using device: {args.device}")

    train_loader, test_loader= load_data(args.batch_size)
    class_names = train_loader.dataset.classes
    results = []

    for model_name in args.models:
        print("\n" + "="*50)
        print(f"Training model: {model_name}")
        print("="*50)

        model = get_model(model_name, args.num_classes).to(args.device)
        checkpoint_path, best_acc = train_model(model, train_loader, test_loader, args.device, args.epochs, args.lr, args.save_dir, model_name)
        model.load_state_dict(torch.load(checkpoint_path))
        acc, precision, recall, f1 = calculate_metrics(model, test_loader, args.device, class_names, args.save_dir, model_name)

        results.append({"Model": model_name, "Best Accuracy": round(best_acc, 4), "Final Accuracy": round(acc, 4), "Precision": round(precision, 4),
            "Recall": round(recall, 4), "F1-Score": round(f1, 4)})

    # Save comparison CSV
    df = pd.DataFrame(results)
    csv_path = "results/benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print("\n Benchmark Results saved to:", csv_path)
    print(df)

# -------------------------------
# Model Loader
# -------------------------------
def get_model(model_name, num_classes):

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "mobilenet_v3s":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "mobilenet_v3l":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == "googlenet":
        model = models.googlenet(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "squeezenet1_1":
        model = models.squeezenet1_1(weights=None)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        model.num_classes = num_classes

    elif model_name == "shufflenet_v2_1":
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "shufflenet_v2_2":
        model = models.shufflenet_v2_x2_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "regnet_400":
        model = models.regnet_y_400mf(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model

# resnet18", "vgg16", "mobilenet_v2", "efficientnet_b0", "alexnet", "densenet121", "
# -------------------------------
# Argument Parser
# -------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Benchmark multiple models on image dataset")
    parser.add_argument("--dataset", type=str, default="GTSRB")
    parser.add_argument("--num_classes", type=int, default=43)
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Path to dataset folder")
    parser.add_argument("--models", nargs="+", default=["resnet18", "mobilenet_v2", "mobilenet_v3s", "mobilenet_v3l", "efficientnet_b0", "densenet121", 
                                                        "googlenet", "squeezenet1_1", "shufflenet_v2_1", "shufflenet_v2_2", "regnet_400"], help="List of models to train")  
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save models and metrics")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    return parser.parse_args()

if __name__ == "__main__":
    main()

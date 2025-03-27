import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb
from utils import find_optimal_batch_size  # Optional: for batch size tuning

################################################################################
# Model Definition: Pretrained ResNet50 for CIFAR-100
################################################################################
class PretrainedResNet50(nn.Module):
    def __init__(self):
        super(PretrainedResNet50, self).__init__()
        # Load a pretrained ResNet50 with ImageNet weights
        self.model = torchvision.models.resnet50(pretrained=True)
        # Replace the final fully-connected layer to output 100 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 100)
        
    def forward(self, x):
        return self.model(x)

################################################################################
# One Epoch Training Function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({
            "loss": running_loss / (total / CONFIG["batch_size"]),
            "acc": 100. * correct / total
        })
    
    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

################################################################################
# Validation Function (CONFIG reference removed from set_postfix)
################################################################################
def validate(model, valloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                "loss": running_loss / (i + 1),
                "acc": 100. * correct / total
            })
    val_loss = running_loss / len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

################################################################################
# Main Function for Part 3
################################################################################
def main():
    CONFIG = {
        "model": "PretrainedResNet50",
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 25,
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
    }
    
    import pprint
    pprint.pprint(CONFIG)
    
    # Set seeds for reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    
    ############################################################################
    # Data Transformations using ImageNet statistics
    ############################################################################
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Load CIFAR-100 datasets: use augmented version for training and non-augmented for validation
    full_train_dataset = torchvision.datasets.CIFAR100(
        root=CONFIG["data_dir"], train=True, download=True, transform=transform_train)
    full_val_dataset = torchvision.datasets.CIFAR100(
        root=CONFIG["data_dir"], train=True, download=True, transform=transform_test)
    
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))  # 20% for validation
    np.random.seed(CONFIG["seed"])
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    
    trainset = torch.utils.data.Subset(full_train_dataset, train_idx)
    valset = torch.utils.data.Subset(full_val_dataset, val_idx)
    
    testset = torchvision.datasets.CIFAR100(
        root=CONFIG["data_dir"], train=False, download=True, transform=transform_test)
    
    ############################################################################
    # Data Loaders
    ############################################################################
    SEARCH_BATCH_SIZES = False
    if SEARCH_BATCH_SIZES:
        dummy_model = PretrainedResNet50().to(CONFIG["device"])
        print("Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(dummy_model, trainset, CONFIG["device"], CONFIG["num_workers"])
        CONFIG["batch_size"] = optimal_batch_size
        print(f"Using batch size: {CONFIG['batch_size']}")
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"],
                                               shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"],
                                             shuffle=False, num_workers=CONFIG["num_workers"])
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"],
                                              shuffle=False, num_workers=CONFIG["num_workers"])
    
    ############################################################################
    # Instantiate Model, Loss Function, Optimizer, and Scheduler
    ############################################################################
    model = PretrainedResNet50().to(CONFIG["device"])
    print("\nModel Summary:")
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"],
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    ############################################################################
    # Initialize Weights & Biases (WandB)
    ############################################################################
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model)
    
    ############################################################################
    # Training Loop
    ############################################################################
    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })
        
        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")
    
    wandb.finish()
    
    # Note: eval_cifar100.py loads the best model from "best_model.pth".
    import eval_cifar100
    import eval_ood
    
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")
    
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()

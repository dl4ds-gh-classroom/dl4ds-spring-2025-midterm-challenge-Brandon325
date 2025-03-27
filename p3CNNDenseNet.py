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
from torchvision.transforms import RandomErasing, RandAugment

################################################################################
# Model Definition: Pretrained DenseNet121 for CIFAR-100
################################################################################
class PretrainedDenseNet121(nn.Module):
    def __init__(self):
        super(PretrainedDenseNet121, self).__init__()
        # Load a pretrained DenseNet121 with ImageNet weights
        self.model = torchvision.models.densenet121(pretrained=True)
        # Replace the classifier (fc layer) to output 100 classes
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, 100)
        
    def forward(self, x):
        return self.model(x)

################################################################################
# Advanced Data Augmentation Functions 
################################################################################
def mixup_data(x, y, alpha=0.4, device='cpu'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0, device='cpu'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

################################################################################
# One Epoch Training Function with Advanced Augmentation
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, scheduler, CONFIG):
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Randomly choose between Mixup and CutMix
        if np.random.random() < 0.5:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0, device=device)
        else:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha=1.0, device=device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        current_step = epoch + (i + 1) / len(trainloader)
        scheduler.step(current_step)
        
        progress_bar.set_postfix({
            "loss": running_loss / (total / CONFIG["batch_size"]),
            "acc": 100. * correct / total
        })
    
    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

################################################################################
# Validation Function
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
# Main Function for Part 3 using DenseNet121 and Advanced Strategies
################################################################################
def main():
    CONFIG = {
        "model": "PretrainedDenseNet121",
        "batch_size": 16,
        "learning_rate": 0.001,
        "epochs": 45,
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
    # Enhanced Data Transformations
    ############################################################################
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
        RandomErasing(p=0.2),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    
    # Load CIFAR-100 datasets
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
        dummy_model = PretrainedDenseNet121().to(CONFIG["device"])
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
    model = PretrainedDenseNet121().to(CONFIG["device"])
    print("\nModel Summary:")
    print(model)
    
    # Label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                           weight_decay=0.01)
    
    # OneCycleLR scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG["learning_rate"],
        epochs=CONFIG["epochs"],
        steps_per_epoch=len(trainloader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
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
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, scheduler, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        # No extra scheduler.step() outside the per-batch update
        
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
    
    # Evaluation: eval_cifar100.py loads the best model from "best_model.pth".
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

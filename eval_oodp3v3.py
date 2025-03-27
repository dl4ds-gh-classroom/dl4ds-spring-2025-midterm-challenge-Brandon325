import torch
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import urllib.request

# CHANGED: Create a custom dataset to apply the same transforms as the test set.
class OODDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None):
        """
        images: numpy array of shape (N, H, W, C) with dtype uint8
        transform: torchvision transform to apply (should match your test transforms)
        """
        self.images = images
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]  # Expecting a numpy array, shape (H, W, C)
        # Convert to PIL image so that transforms can be applied
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        return image

def evaluate_ood(model, distortion_name, severity, CONFIG):
    data_dir = CONFIG["ood_dir"]
    device = CONFIG["device"]

    # Load the OOD images (assumed to be in uint8 format)
    images = np.load(os.path.join(data_dir, f"{distortion_name}.npy"))
    
    # Select the subset of images for the given severity
    start_index = (severity - 1) * 10000
    end_index = severity * 10000
    images = images[start_index:end_index]
    
    # CHANGED: Ensure images are in uint8 format instead of converting immediately to a tensor.
    images = images.astype(np.uint8)
    
    # CHANGED: Define a transform pipeline matching your test transforms.
    # This ensures that each OOD image is resized to 256, center cropped to 224,
    # converted to a tensor, and normalized using ImageNet statistics.
    ood_transform = transforms.Compose([
        transforms.Resize(256),              # Same as in your test transforms in p3CNNv2.py
        transforms.CenterCrop(224),          # Same as in your test transforms in p3CNNv2.py
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),  # ImageNet normalization
                             (0.229, 0.224, 0.225))
    ])
    
    # Create the dataset and DataLoader using the custom dataset.
    dataset = OODDataset(images, transform=ood_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=CONFIG["num_workers"], 
        pin_memory=True)
    
    predictions = []  # Store predictions
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc=f"Evaluating {distortion_name} (Severity {severity})", leave=False):
           inputs = inputs.to(device)
           outputs = model(inputs)
           _, predicted = outputs.max(1)
           predictions.extend(predicted.cpu().numpy())
    return predictions

# No changes here.
def files_already_downloaded(directory, num_files):
    for i in range(num_files):
        file_name = f"distortion{i:02d}.npy"
        file_path = os.path.join(directory, file_name)
        if not os.path.exists(file_path):
            return False
    return True

def evaluate_ood_test(model, CONFIG):
    data_dir = CONFIG["ood_dir"]
    device = CONFIG["device"]

    num_files = 19  # Number of files to download

    # Only download if files aren't already downloaded
    if not files_already_downloaded(data_dir, num_files):
        os.makedirs(data_dir, exist_ok=True)
        base_url = "https://github.com/DL4DS/ood-test-files/raw/refs/heads/main/ood-test/"
        for i in range(num_files):
            file_name = f"distortion{i:02d}.npy"
            file_url = base_url + file_name
            file_path = os.path.join(data_dir, file_name)
            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(file_url, file_path)
            print(f"Downloaded {file_name} to {file_path}")
        print("All files downloaded successfully.")
    else:
        print("All files are already downloaded.")

    distortions = [f"distortion{str(i).zfill(2)}" for i in range(19)]
    all_predictions = []  # Store all predictions for the submission file

    model.eval()  # Ensure model is in evaluation mode
    for distortion in distortions:
        for severity in range(1, 6):
            preds = evaluate_ood(model, distortion, severity, CONFIG)
            all_predictions.extend(preds)  # Accumulate predictions
            print(f"{distortion} (Severity {severity}) evaluated.")
    return all_predictions

def create_ood_df(all_predictions):
    distortions = [f"distortion{str(i).zfill(2)}" for i in range(19)]
    # Create IDs for OOD (assuming the order is as evaluated)
    ids_ood = []
    for distortion in distortions:
        for severity in range(1, 6):
            for i in range(10000):
              ids_ood.append(f"{distortion}_{severity}_{i}")
    submission_df_ood = pd.DataFrame({'id': ids_ood, 'label': all_predictions})
    return submission_df_ood

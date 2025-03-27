import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader

# Import your evaluation modules
import eval_cifar100
import eval_ood

# If your model is defined in a separate module, you could import it instead.
# from your_model_module import PretrainedResNet50

################################################################################
# Model Definition: Pretrained ResNet50 (copy if not imported)
################################################################################
class PretrainedResNet50(torch.nn.Module):
    def __init__(self):
        super(PretrainedResNet50, self).__init__()
        # Load ResNet50 with pretrained ImageNet weights
        self.model = models.resnet50(pretrained=True)
        # Replace the final fully-connected layer to match CIFAR-100 (100 classes)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 100)

    def forward(self, x):
        return self.model(x)

################################################################################
# CONFIG dictionary (ensure it matches what you used during training)
################################################################################
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir": "./data",  # update as needed
}

################################################################################
# Prepare the Test Data Loader
################################################################################
# Define test transforms (must be same as used during training evaluation)
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load the CIFAR-100 test dataset
testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False,
                                        download=True, transform=transform_test)

testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

################################################################################
# Load the best model and run evaluation
################################################################################
model = PretrainedResNet50().to(CONFIG["device"])
model.load_state_dict(torch.load("best_model.pth", map_location=CONFIG["device"]))

# Evaluate on CIFAR-100 Test Set
predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

# Evaluate OOD if required
all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
submission_df_ood = eval_ood.create_ood_df(all_predictions)
submission_df_ood.to_csv("submission_ood_p3v2.csv", index=False)
print("submission_ood_p3v2.csv created successfully.")

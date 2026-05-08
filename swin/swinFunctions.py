import os
import re
import pandas as pd
import random
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import torch


def parse_breakhis_dataset(data_root):
    """
    Traverses the BreaKHis directory structure and returns a pandas DataFrame.
    """
    records = []

    # b / m
    for tumor_type in ["benign", "malignant"]:
        tumor_dir = os.path.join(data_root, tumor_type)
        if not os.path.exists(tumor_dir):
            continue

        # under SOB folder
        sob_dir = os.path.join(tumor_dir, "SOB")
        if not os.path.exists(sob_dir):
            continue

        # Traverse subtype (eg ductal_carcinoma, tubular_adenoma)
        for subtype in os.listdir(sob_dir):
            subtype_dir = os.path.join(sob_dir, subtype)
            if not os.path.isdir(subtype_dir):
                continue

            # Traverse subject (eg SOB_M_DC-14-10926)
            for subject_folder in os.listdir(subtype_dir):
                subject_dir = os.path.join(subtype_dir, subject_folder)
                if not os.path.isdir(subject_dir):
                    continue

                # extract subject_id → "<YEAR>-<SLIDE_ID>"
                m = re.search(r'(?P<year>\d{2})[-_](?P<slide>\d+[A-Z]{0,3})$', subject_folder)
                subject_id = f"{m.group('year')}-{m.group('slide')}" if m else subject_folder

                # Traverse magnification (40X, 100X, 200X, 400X)
                for mag_folder in os.listdir(subject_dir):
                    mag_dir = os.path.join(subject_dir, mag_folder)
                    if not os.path.isdir(mag_dir):
                        continue

                    # extract magnification
                    try:
                        magnification = int(''.join(filter(str.isdigit, mag_folder)))
                    except:
                        magnification = None

                    # Traverse image file
                    for fname in os.listdir(mag_dir):
                        if not fname.lower().endswith(".png"):
                            continue

                        fpath = os.path.join(mag_dir, fname)

                        # extract tumor_type (b/m)
                        name_parts = fname.split('_')
                        tumor_flag = name_parts[1].lower() if len(name_parts) > 1 else tumor_type[0]

                        # subtype abbreviation
                        subtype_abbr = name_parts[2].split('-')[0].lower() if len(name_parts) > 2 else subtype.lower()

                        # verify magnification
                        mag_in_name = None
                        if '-' in fname:
                            segs = fname.split('-')
                            for s in segs:
                                if s.isdigit() and s in ["40", "100", "200", "400"]:
                                    mag_in_name = int(s)
                                    break
                        magnification = mag_in_name or magnification

                        records.append({
                            "filepath": fpath,
                            "tumor_type": tumor_flag,
                            "subtype": subtype_abbr,
                            "magnification": magnification,
                            "subject_id": subject_id
                        })

    return pd.DataFrame(records)

def save_metadata(df, data_root, filename="BreaKHis_metadata.csv"):
    """
    Saves the DataFrame to a CSV file in the data_root directory.
    """
    save_path = os.path.join(data_root, filename)
    df.to_csv(save_path, index=False)
    print(f"✅ Metadata CSV saved to: {save_path}")
    return save_path


def split_data_by_subject(input_path, output_path, train_ratio=0.8, seed=42):
    """
    Splits the dataset into train and validation sets ensuring that images 
    from the same subject stay in the same partition.
    """
    random.seed(seed)
    
    # Load data
    data = pd.read_csv(input_path)

    # Identify and shuffle unique subjects
    unique_subjects = data["subject_id"].unique().tolist()
    random.shuffle(unique_subjects)

    # Calculate split index
    n = len(unique_subjects)
    n_train = int(train_ratio * n)
    
    train_subjects = set(unique_subjects[:n_train])
    val_subjects = set(unique_subjects[n_train:])

    # Helper function for assignment
    def assign_partition(sid):
        if sid in train_subjects:
            return "train"
        elif sid in val_subjects:
            return "val"
        return None

    # Apply assignment using the column directly (faster than row apply)
    data["partition"] = data["subject_id"].map(assign_partition)

    # Save and return
    data.to_csv(output_path, index=False)
    print(f"✅ Split done. Data saved to: {output_path}")
    print(f"   Total subjects: {n} (Train: {len(train_subjects)}, Val: {len(val_subjects)})")
    
    return data

import torch

def get_training_config(custom_root=None):
    """
    Returns the standard hyperparameters and device configuration.
    """
    # Use a custom path if provided, otherwise default to your path
    root = custom_root if custom_root else 'C:/Users/amana/Downloads/BreaKHis_v1/histology_slides/breast'
    
    config = {
        "METADATA_PATH": os.path.join(root, 'BreaKHis_metadata_split.csv'),
        "NUM_CLASSES": 2,
        "BATCH_SIZE": 16,
        "LEARNING_RATE": 1e-4,
        "NUM_EPOCHS": 5,
        "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    }
    
    print(f"✅ Configuration set. Using device: {config['DEVICE']}")
    return config

class BreaKHisDataset(Dataset):
    """A custom PyTorch Dataset for loading BreaKHis images."""
    def __init__(self, metadata_df: pd.DataFrame, transform=None):
        self.metadata = metadata_df
        self.transform = transform
        self.label_map = {'b': 0, 'm': 1}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.metadata.iloc[idx]['filepath']
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Could not open image {img_path}: {e}")
            return None, None 

        raw_label = self.metadata.iloc[idx]['tumor_type']
        label = self.label_map.get(raw_label, -1)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    """Helper function to remove corrupt samples from DataLoader."""
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

def get_transforms(img_size=224):
    """Defines the standard image transformations."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def create_dataloaders(df, train_transform, val_transform, batch_size=16):
    """Splits the metadata and creates the DataLoaders."""
    train_df = df[df['partition'] == 'train'].reset_index(drop=True)
    val_df = df[df['partition'] == 'val'].reset_index(drop=True)

    train_dataset = BreaKHisDataset(train_df, transform=train_transform)
    val_dataset = BreaKHisDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, collate_fn=collate_fn)
    
    print(f"✅ DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_loader, val_loader

def load_swin_transformer(num_classes, device):
    """
    Loads a pre-trained Swin Transformer from the timm library and adapts it 
    for the binary classification task.
    """
    # Create the model using timm
    model = timm.create_model(
        'swin_tiny_patch4_window7_224', 
        pretrained=True, 
        num_classes=num_classes
    )
    
    print(f"✅ Loaded Swin Transformer ('swin_tiny_patch4_window7_224') pre-trained on ImageNet.")
    
    # Move model to the specified device (cuda or cpu)
    return model.to(device)

from tqdm import tqdm
import torch

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, device):
    """Main function to train and validate the model."""
    
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        print(f"\n--- Epoch {epoch+1}/{num_epochs} (Training) ---")
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False)
        
        for images, labels in train_pbar:
            if images is None:
                continue
                
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        print(f"--- Epoch {epoch+1}/{num_epochs} (Validation) ---")
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False)
        
        with torch.no_grad():
            for images, labels in val_pbar:
                if images is None:
                    continue
                    
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_swin_model.pth')
            print("Model saved to 'best_swin_model.pth' based on improved validation accuracy.")
        
    print("\nTraining complete.")

    import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, val_loader, device):
    """
    Performs a full evaluation on the validation set, returns labels/preds, 
    and prints a classification report.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print("--- Starting Comprehensive Evaluation ---")
    with torch.no_grad():
        for images, labels in val_loader:
            if images is None:
                continue
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Get probabilities for AUC
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_probs.extend(probs[:, 1].cpu().numpy()) # Probability of class 1 (Malignant)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate Metrics
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))
    
    auc_score = roc_auc_score(all_labels, all_probs)
    print(f"Validation AUC: {auc_score:.4f}")

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(labels, preds, title="Confusion Matrix"):
    """Plots a heatmap of the confusion matrix."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'], 
                yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

def plot_roc_curve(labels, probs):
    """Plots the ROC curve and calculates AUC."""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def evaluate_model(model, data_loader, set_name, device):
    """
    Calculates all key metrics for a given data loader, prints a report, 
    and displays CM and ROC plots.
    """
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"Evaluating {set_name}"):
            if images is None:
                continue
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Probability of Malignant (class 1)
            probabilities = F.softmax(outputs, dim=1)[:, 1] 
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # 1. Classification Report
    print(f"\n--- Classification Report for {set_name} ---")
    print(classification_report(all_labels, all_predictions, target_names=['Benign (0)', 'Malignant (1)']))

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Benign', 'Predicted Malignant'], 
                yticklabels=['Actual Benign', 'Actual Malignant'])
    plt.title(f'Confusion Matrix - {set_name}')
    plt.show()

    # 3. ROC AUC Score
    roc_auc = roc_auc_score(all_labels, all_probabilities)
    print(f"\nROC AUC Score for {set_name}: {roc_auc:.4f}")

    # 4. Plot ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {set_name}')
    plt.legend(loc="lower right")
    plt.show()

    return cm, roc_auc, fpr, tpr


from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def perform_patient_level_analysis(model, data_loader, metadata_df, device):
    """
    Aggregates image-level predictions into patient-level results using majority voting.
    """
    # Filter metadata to only include validation subjects to match the DataLoader
    val_metadata_df = metadata_df[metadata_df['partition'] == 'val'].reset_index(drop=True)
    
    all_predictions = []
    all_true_labels = []
    all_subject_ids = []
    
    model.eval()
    with torch.no_grad():
        # We assume the DataLoader is NOT shuffled for evaluation to maintain order
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="Collecting Patient Data")):
            if images is None:
                continue
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Map batch back to original metadata indices
            start_idx = batch_idx * data_loader.batch_size
            for i in range(len(labels)):
                data_index = start_idx + i
                if data_index < len(val_metadata_df):
                    all_subject_ids.append(val_metadata_df.iloc[data_index]['subject_id'])
                    all_true_labels.append(labels[i].item())
                    all_predictions.append(predicted[i].item())

    # Create results dataframe
    results_df = pd.DataFrame({
        'subject_id': all_subject_ids,
        'true_label': all_true_labels,
        'image_prediction': all_predictions
    })
    
    # Aggregate by subject (Majority Voting)
    # Mean > 0.5 means more than half the images were predicted as Malignant (1)
    patient_stats = results_df.groupby('subject_id').agg({
        'true_label': 'first',
        'image_prediction': 'mean'
    }).reset_index()
    
    patient_stats.rename(columns={'image_prediction': 'malignant_ratio'}, inplace=True)
    patient_stats['patient_prediction'] = (patient_stats['malignant_ratio'] > 0.5).astype(int)

    # 4. Metrics & Visualization
    y_true = patient_stats['true_label'].values
    y_pred = patient_stats['patient_prediction'].values

    print("\n" + "="*30)
    print(" PATIENT-LEVEL EVALUATION ")
    print("="*30)
    print(classification_report(y_true, y_pred, target_names=['Benign Patient', 'Malignant Patient']))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Pred Benign', 'Pred Malignant'], 
                yticklabels=['Actual Benign', 'Actual Malignant'])
    plt.title('Patient-Level Confusion Matrix (Majority Vote)')
    plt.show()

    accuracy = (y_true == y_pred).mean()
    print(f"Final Patient-Level Accuracy: {accuracy * 100:.2f}%")
    
    return patient_stats
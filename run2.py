# Import Libraries
import os
import glob
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import foolbox as fb
from foolbox.attacks import FGSM, PGD, LinfBasicIterativeAttack, BoundaryAttack, L2DeepFoolAttack, L2CarliniWagnerAttack
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import zipfile
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchsummary import summary
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
import zipfile
import random


print("Loading and preprocessing data...")

df=pd.read_csv("merged_IDS2018.csv")

# Create Class column
df['Label'] = df['Label'].str.strip()
df['Class'] = df['Label'].apply(lambda x: 'Benign' if x == 'Benign' else 'Attack')

# Identify columns where every single row has the same value
constant_cols = [col for col in df.columns if df[col].nunique() <= 1]

# Drop them safely
df.drop(columns=constant_cols, inplace=True)

# columns to drop
columns_to_drop = ['Dst Port', 
                   'Flow Byts/s', 'Flow Pkts/s', 'Fwd Pkts/s', 'Bwd Pkts/s',
                   'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg']
df.drop(columns=columns_to_drop, inplace=True)

# remove the outliers (negative values) from selected columns
ngtv_cols = ['Flow Duration','Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
             'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
             'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min']
for col in ngtv_cols:
    df = df[df[col] >= 0]

# Cap the upper limit of selected columns to their 95th percentile
for i in ['Flow Duration', 
          'Tot Fwd Pkts', 'Tot Bwd Pkts', 
          'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 
          'Fwd Pkt Len Max', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
          'Bwd Pkt Len Max', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std',
          'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
          'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
          'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
          'Fwd Header Len', 'Bwd Header Len',
          'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
          'Down/Up Ratio',
          'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
          'Fwd Act Data Pkts',
          'Active Mean', 'Active Std', 'Active Max', 'Active Min',
           'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
          ]:
    upper_limit = df[i].quantile(0.95)
    df[i] = df[i].clip(upper=upper_limit)

# Split features and targets
X = df.drop(columns=['Label', 'Class'])
y_binary = df['Class']
# y_multi = df['Label']

def prepare_data(X, y, test_size=0.2, random_state=42):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=test_size,random_state=random_state,stratify=y_binary
        )
    
    # Split the remaining 20% into Test (50% total) and Validation (50% total)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test,
        test_size=0.50,
        random_state=random_state,
        stratify=y_test
        )

    ros = RandomOverSampler(
        sampling_strategy='auto',  
        random_state=random_state
        )
    
    X_train, y_train = ros.fit_resample(X_train, y_train)
    
    # Define column groups
    categorical_cols = [
        'Protocol',
        'Fwd PSH Flags','Fwd URG Flags',
        'FIN Flag Cnt','SYN Flag Cnt','RST Flag Cnt','ACK Flag Cnt',
        'URG Flag Cnt','PSH Flag Cnt','ECE Flag Cnt','CWE Flag Count'
        ]
    
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    # Build the transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
      ]
    )

    # Apply the transformations and cast to float32 
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    # Encode the labels 
    le_label = LabelEncoder()
    y_train = le_label.fit_transform(y_train)
    y_val = le_label.transform(y_val)
    y_test = le_label.transform(y_test)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    X_val = torch.FloatTensor(X_val)

    y_train = torch.FloatTensor(y_train)  
    y_val = torch.FloatTensor(y_val)      
    y_test = torch.LongTensor(y_test) 
   
    return X_train, X_test, X_val, y_train, y_test, y_val, preprocessor, le_label

X_train, X_test, X_val, y_train, y_test, y_val, preprocessor, le_label = prepare_data(X, y_binary)


class DDoSDetector(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        layers = []
        in_channels = 1
        for _ in range(10):
            layers.append(nn.Conv1d(in_channels, 108, kernel_size=5, padding=2))
            layers.append(nn.BatchNorm1d(108))
            layers.append(nn.ReLU())
            in_channels = 108
        self.cnn_backbone = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(108, num_classes)

    def forward(self, x, return_features=False):
        x = x.unsqueeze(1)
        x = self.cnn_backbone(x)
        features = self.global_pool(x).squeeze(-1)  # (B, 108)
        logit = self.classifier(features)            # (B, 1)
        if return_features:
            return logit, features
        return logit

# Wrapper to convert binary classifier output to 2-class logits for foolbox
class DDoSDetectorBinaryWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        logit = self.model(x)
        zeros = torch.zeros_like(logit)
        return torch.cat([zeros, logit], dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fgsm_attack  = FGSM()
ifgsm_attack = LinfBasicIterativeAttack(abs_stepsize=0.01, steps=20)
pgd_attack   = PGD(rel_stepsize=0.02, steps=40, random_start=True)

attack_pool  = [fgsm_attack, ifgsm_attack, pgd_attack]
epsilon_range = [0.03, 0.07, 0.1]

# Single model — no teachers
union_model = DDoSDetector(num_classes=1).to(device)
union_model.load_state_dict(torch.load('baseline_nids_cnn.pth', map_location=device))

criterion = nn.BCEWithLogitsLoss()
optimizer  = optim.Adam(union_model.parameters(), lr=1e-3, weight_decay=1e-5)

# Foolbox wrapper for the union model itself
def get_fmodel(model):
    wrapped = DDoSDetectorBinaryWrapper(model).to(device)
    wrapped.eval()
    return fb.PyTorchModel(wrapped, bounds=(0.0, 1.0))

epochs = 10

# Split into clean and adversarial pools like your teacher training
indices = torch.randperm(len(X_train))
split   = len(X_train) // 2

clean_idx = indices[:split]
adv_idx   = indices[split:]

X_clean, y_clean     = X_train[clean_idx], y_train[clean_idx]
X_adv_pool, y_adv_pool = X_train[adv_idx],   y_train[adv_idx]

train_clean_loader = DataLoader(TensorDataset(X_clean, y_clean),       batch_size=1024, shuffle=True)
train_adv_loader   = DataLoader(TensorDataset(X_adv_pool, y_adv_pool), batch_size=1024, shuffle=True)
val_loader         = DataLoader(TensorDataset(X_val, y_val),           batch_size=1024, shuffle=False)
test_loader        = DataLoader(TensorDataset(X_test, y_test),         batch_size=1024, shuffle=False)

for epoch in range(epochs):
    union_model.train()
    running_loss = 0.0

    pbar = tqdm(zip(train_clean_loader, train_adv_loader),
                desc=f"Epoch {epoch+1}/{epochs}",
                total=min(len(train_clean_loader), len(train_adv_loader)))

    for (clean_inputs, clean_labels), (adv_inputs, adv_labels) in pbar:
        clean_inputs = clean_inputs.to(device)
        clean_labels = clean_labels.to(device)
        adv_inputs   = adv_inputs.to(device)
        adv_labels   = adv_labels.to(device)

        # Randomly pick one attack and one epsilon this batch
        attack  = random.choice(attack_pool)
        epsilon = random.choice(epsilon_range)

        # Generate adversarial examples
        union_model.eval()
        is_malicious = (adv_labels == 0)
        adv_inputs_perturbed = adv_inputs.clone()

        if is_malicious.any():
            X_mal = adv_inputs[is_malicious]
            y_mal = adv_labels[is_malicious].long()

            fmodel = get_fmodel(union_model)

            with torch.enable_grad():
                _, adv_mal, _ = attack(fmodel, X_mal, y_mal, epsilons=epsilon)

            adv_mal = torch.clamp(adv_mal, 0.0, 1.0)
            adv_inputs_perturbed[is_malicious] = adv_mal

        # Combine clean + adversarial
        mixed_inputs = torch.cat([clean_inputs, adv_inputs_perturbed], dim=0)
        mixed_labels = torch.cat([clean_labels, adv_labels], dim=0)

        # Update model
        union_model.train()
        optimizer.zero_grad()
        outputs = union_model(mixed_inputs).squeeze(-1)
        loss    = criterion(outputs, mixed_labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}",
                          'attack': attack.__class__.__name__,
                          'eps': f"{epsilon:.2f}"})

    avg_loss = running_loss / min(len(train_clean_loader), len(train_adv_loader))

    # Validation
    union_model.eval()
    all_preds, all_labels_list = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = union_model(inputs).squeeze(-1)
            preds   = (torch.sigmoid(outputs) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())

    val_acc = accuracy_score(all_labels_list, all_preds) * 100
    val_f1  = f1_score(all_labels_list, all_preds)
    print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | Val Acc {val_acc:.2f}% | Val F1 {val_f1:.4f}")

torch.save(union_model.state_dict(), 'union_adv_model.pth')
print("Union adversarial model saved.")

# Final test
union_model.eval()
all_preds, all_labels_list = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = union_model(inputs).squeeze(-1)
        preds   = (torch.sigmoid(outputs) > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels_list.extend(labels.cpu().numpy())

print("\n--- UNION ADVERSARIAL TRAINING TEST METRICS (Clean) ---")
print(classification_report(all_preds, all_labels_list, target_names=['Attack', 'Benign']))
# store in json file
results = {
    'union_adv_test': {
        'accuracy': accuracy_score(all_labels_list, all_preds),
        'precision': precision_score(all_labels_list, all_preds),
        'recall': recall_score(all_labels_list, all_preds),
        'f1_score': f1_score(all_labels_list, all_preds),
        'report': classification_report(all_preds, all_labels_list, target_names=['Attack', 'Benign'], output_dict=True)
    }
}
import json
with open('union_adv_results.json', 'w') as f:
    json.dump(results, f, indent=4)
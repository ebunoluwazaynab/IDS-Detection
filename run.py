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


def compute_adaptive_weights(student_logits, teacher_logits_list):
    similarities = []
    for t_logits in teacher_logits_list:
        sim = F.cosine_similarity(student_logits, t_logits, dim=0).mean()
        similarities.append(sim)

    weights = [1.0 + s for s in similarities]
    total = sum(weights)
    normalized = [w / total for w in weights]
    return normalized


def distillation_loss(student_logits, teacher_logits_list, weights, temperature=4.0):
    s_soft = torch.sigmoid(student_logits / temperature)
    student_probs = torch.cat([1 - s_soft, s_soft], dim=1)
    student_log_probs = torch.log(student_probs.clamp(min=1e-8))

    weighted_t_soft = torch.zeros_like(s_soft)
    for t_logits, w in zip(teacher_logits_list, weights):
        t_soft = torch.sigmoid(t_logits / temperature)
        weighted_t_soft += w * t_soft
    teacher_probs = torch.cat([1 - weighted_t_soft, weighted_t_soft], dim=1)

    kld = F.kl_div(student_log_probs, teacher_probs.clamp(min=1e-8), reduction='batchmean')
    return kld * (temperature ** 2)


def feature_distillation_loss(student_features, teacher_features_list, weights):
    loss = torch.tensor(0.0, device=student_features.device)
    for t_features, w in zip(teacher_features_list, weights):
        loss += w * F.mse_loss(student_features, t_features.detach())
    return loss

print("Setting up teachers and attacks...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fgsm_teacher = DDoSDetector(num_classes=1).to(device)
fgsm_teacher.load_state_dict(torch.load('fgsm_teacher_model.pth', map_location=device))
fgsm_teacher.eval()

ifgsm_teacher = DDoSDetector(num_classes=1).to(device)
ifgsm_teacher.load_state_dict(torch.load('ifgsm_teacher_model.pth', map_location=device))
ifgsm_teacher.eval()

pgd_teacher = DDoSDetector(num_classes=1).to(device)
pgd_teacher.load_state_dict(torch.load('pgd_teacher_model.pth', map_location=device))
pgd_teacher.eval()

teachers = [fgsm_teacher, ifgsm_teacher, pgd_teacher]


print("Precomputing foolbox models and setting up attacks...")

fgsm_attack  = FGSM()
ifgsm_attack = LinfBasicIterativeAttack(abs_stepsize=0.01, steps=20)
pgd_attack   = PGD(rel_stepsize=0.02, steps=40, random_start=True)
attack_list  = [fgsm_attack, ifgsm_attack, pgd_attack]
epsilon_range = [0.03, 0.07, 0.1]

# Precompute foolbox models for each teacher
fmodels = []
for teacher in teachers:
    wrapped = DDoSDetectorBinaryWrapper(teacher).to(device)
    wrapped.eval()
    fmodel = fb.PyTorchModel(wrapped, bounds=(0.0, 1.0))
    fmodels.append(fmodel)

student = DDoSDetector(num_classes=1).to(device)
criterion_ce = nn.BCEWithLogitsLoss()
optimizer    = optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-5)


print("Starting MTKD-ADR training...")
alpha  = 0.4   
beta   = 0.4  
epochs = 10

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=1024, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=1024, shuffle=False)

for epoch in range(epochs):
    student.train()
    for t in teachers:
        t.eval()

    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        is_malicious = (labels == 0)
        epsilon = random.choice(epsilon_range)

        # Generate adversarial batch per attack/teacher pair
        adv_batches = []
        for attack, fmodel_t in zip(attack_list, fmodels):
            adv_batch = inputs.clone()
            if is_malicious.sum() > 0:
                X_mal = inputs[is_malicious]
                y_mal = labels[is_malicious]
                with torch.enable_grad():
                    _, adv_mal, _ = attack(fmodel_t, X_mal, y_mal.long(), epsilons=epsilon)
                adv_mal = torch.clamp(adv_mal, 0.0, 1.0)
                adv_batch = inputs.clone()
                adv_batch[is_malicious] = adv_mal
            adv_batches.append(adv_batch)

        # Stack: clean + fgsm_adv + ifgsm_adv + pgd_adv
        all_inputs = torch.cat([inputs] + adv_batches, dim=0)   # (4B, features)
        all_labels = torch.cat([labels] * 4, dim=0)             # (4B,)

        optimizer.zero_grad()

        # Student forward — get both logits and features
        student_logits, student_features = student(all_inputs, return_features=True)

        # Teacher forwards
        with torch.no_grad():
            teacher_logits_list   = []
            teacher_features_list = []
            for t in teachers:
                t_logits, t_features = t(all_inputs, return_features=True)
                teacher_logits_list.append(t_logits)
                teacher_features_list.append(t_features)

        # Adaptive weights from logits
        weights = compute_adaptive_weights(student_logits.detach(), teacher_logits_list)

        # Losses
        ce_loss      = criterion_ce(student_logits.squeeze(-1), all_labels.float())
        kd_loss      = distillation_loss(student_logits, teacher_logits_list, weights)
        feat_loss    = feature_distillation_loss(student_features, teacher_features_list, weights)
        total_loss   = (1 - alpha - beta) * ce_loss + alpha * kd_loss + beta * feat_loss

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        pbar.set_postfix({'loss':  f"{total_loss.item():.4f}",
                          'ce':    f"{ce_loss.item():.4f}",
                          'kd':    f"{kd_loss.item():.4f}",
                          'feat':  f"{feat_loss.item():.4f}",
                          'eps':   f"{epsilon:.2f}"})

    avg_loss = running_loss / len(train_loader)

    # Diagnostics
    student.eval()
    diag_inputs, _ = next(iter(train_loader))
    diag_inputs = diag_inputs.to(device)

    with torch.no_grad():
        s_logits, s_features = student(diag_inputs, return_features=True)
        t_logits_list, t_features_list = [], []
        for t in teachers:
            tl, tf = t(diag_inputs, return_features=True)
            t_logits_list.append(tl)
            t_features_list.append(tf)

        T = 4.0
        s_soft = torch.sigmoid(s_logits / T)
        student_probs = torch.cat([1 - s_soft, s_soft], dim=1)
        student_log_probs = torch.log(student_probs.clamp(min=1e-8))

        w = compute_adaptive_weights(s_logits, t_logits_list)
        weighted_t_soft = torch.zeros_like(s_soft)
        for tl, wi in zip(t_logits_list, w):
            weighted_t_soft += wi * torch.sigmoid(tl / T)
        teacher_probs = torch.cat([1 - weighted_t_soft, weighted_t_soft], dim=1)
        kld = F.kl_div(student_log_probs, teacher_probs.clamp(min=1e-8), reduction='batchmean')

        # Feature similarity between student and each teacher
        feat_sims = []
        for tf in t_features_list:
            sim = F.cosine_similarity(s_features, tf, dim=1).mean().item()
            feat_sims.append(round(sim, 4))

        print(f"\n── Epoch {epoch+1} Diagnostics ──")
        print(f"Student logits:  {s_logits.min().item():.2f} to {s_logits.max().item():.2f}")
        print(f"Student soft:    {s_soft.min().item():.3f} to {s_soft.max().item():.3f}")
        for i, tl in enumerate(t_logits_list):
            t_soft = torch.sigmoid(tl / T)
            print(f"Teacher {i} logits: {tl.min().item():.2f} to {tl.max().item():.2f} | soft: {t_soft.min().item():.3f} to {t_soft.max().item():.3f}")
        print(f"Adaptive weights: {[round(wi.item(), 3) for wi in w]}")
        print(f"KLD (student→weighted teacher): {kld.item():.4f}")
        for i, tl in enumerate(t_logits_list):
            t_soft_i = torch.sigmoid(tl / T)
            t_probs_i = torch.cat([1 - t_soft_i, t_soft_i], dim=1)
            kld_i = F.kl_div(student_log_probs, t_probs_i.clamp(min=1e-8), reduction='batchmean')
            print(f"KLD (student→teacher {i}): {kld_i.item():.4f}")
        for i in range(len(t_logits_list)):
            for j in range(i+1, len(t_logits_list)):
                ti_soft = torch.sigmoid(t_logits_list[i] / T)
                tj_soft = torch.sigmoid(t_logits_list[j] / T)
                ti_probs = torch.cat([1 - ti_soft, ti_soft], dim=1)
                tj_probs = torch.cat([1 - tj_soft, tj_soft], dim=1)
                ti_log = torch.log(ti_probs.clamp(min=1e-8))
                kld_ij = F.kl_div(ti_log, tj_probs.clamp(min=1e-8), reduction='batchmean')
                print(f"KLD (teacher {i}→teacher {j}): {kld_ij.item():.4f}")
        t_preds = [(torch.sigmoid(tl) > 0.5).long() for tl in t_logits_list]
        agree_all = (t_preds[0] == t_preds[1]) & (t_preds[1] == t_preds[2])
        print(f"Teacher agreement (all 3): {agree_all.float().mean().item()*100:.1f}%")
        majority = (t_preds[0].float() + t_preds[1].float() + t_preds[2].float()) >= 2
        s_preds = (torch.sigmoid(s_logits) > 0.5).long()
        student_majority_agree = (s_preds.squeeze() == majority.squeeze()).float().mean().item()
        print(f"Student vs majority vote agreement: {student_majority_agree*100:.1f}%")
        print(f"Weighted teacher soft mean: {weighted_t_soft.mean().item():.3f}")
        print(f"Weighted teacher soft std:  {weighted_t_soft.std().item():.3f}")
        print(f"Feature cosine sim (student→teachers): {feat_sims}")

    # Validation
    all_preds, all_labels_list = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = student(inputs).squeeze(-1)
            preds  = (torch.sigmoid(logits) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())

    val_acc = accuracy_score(all_labels_list, all_preds) * 100
    val_f1  = f1_score(all_labels_list, all_preds)
    print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | Val Acc {val_acc:.2f}% | Val F1 {val_f1:.4f}")

torch.save(student.state_dict(), 'mtkd_adr_student.pth')
print("Student model saved.")

# Final test
student.eval()
all_preds, all_labels_list = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = student(inputs).squeeze(-1)
        preds  = (torch.sigmoid(logits) > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels_list.extend(labels.cpu().numpy())

print("\n--- MTKD-ADR STUDENT TEST METRICS (Clean) ---")
print(classification_report(all_labels_list, all_preds, target_names=['Attack', 'Benign']))
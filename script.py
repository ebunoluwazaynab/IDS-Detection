# ─────────────────────────────────────────
# Imports
# ─────────────────────────────────────────
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import foolbox as fb
from foolbox.attacks import FGSM, PGD, LinfBasicIterativeAttack
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report)
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 1024
EPOCHS        = 10
LR            = 1e-3
L2_LAMBDA     = 1e-5
L1_LAMBDA     = 1e-6
MAPE_THRESHOLD = 20.0
EPSILON_RANGE  = [0.03, 0.07, 0.1]
EVAL_EPSILONS  = [0.03, 0.07, 0.1]

FEATURES_TO_PERTURB = [
    'Flow Duration',
    'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min', 'Flow IAT Std',
    'Fwd IAT Tot',   'Fwd IAT Mean', 'Fwd IAT Std',  'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Tot',   'Bwd IAT Mean', 'Bwd IAT Std',  'Bwd IAT Max', 'Bwd IAT Min',
    'Active Mean',   'Active Std',   'Active Max',   'Active Min',
    'Idle Mean',     'Idle Std',     'Idle Max',     'Idle Min',
]

CATEGORICAL_COLS = [
    'Protocol',
    'Fwd PSH Flags', 'Fwd URG Flags',
    'FIN Flag Cnt',  'SYN Flag Cnt', 'RST Flag Cnt', 'ACK Flag Cnt',
    'URG Flag Cnt',  'PSH Flag Cnt', 'ECE Flag Cnt', 'CWE Flag Count',
]


# ─────────────────────────────────────────
# Helpers — folder / json
# ─────────────────────────────────────────
def make_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


# ─────────────────────────────────────────
# Data loading and preprocessing
# ─────────────────────────────────────────
def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df['Label'] = df['Label'].str.strip()
    df['Class'] = df['Label'].apply(lambda x: 'Benign' if x == 'Benign' else 'Attack')

    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    df.drop(columns=constant_cols, inplace=True)

    drop_cols = ['Dst Port', 'Flow Byts/s', 'Flow Pkts/s',
                 'Fwd Pkts/s', 'Bwd Pkts/s',
                 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    ngtv_cols = ['Flow Duration', 'Flow IAT Mean', 'Flow IAT Std',
                 'Flow IAT Max', 'Flow IAT Min',
                 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
                 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min']
    for col in ngtv_cols:
        if col in df.columns:
            df = df[df[col] >= 0]

    cap_cols = [
        'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
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
        'Idle Mean',   'Idle Std',   'Idle Max',   'Idle Min',
    ]
    for col in cap_cols:
        if col in df.columns:
            df[col] = df[col].clip(upper=df[col].quantile(0.95))

    X        = df.drop(columns=['Label', 'Class'])
    y_binary = df['Class']

    numerical_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    preprocessor = ColumnTransformer(transformers=[
        ('num', MinMaxScaler(), numerical_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), CATEGORICAL_COLS),
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_val   = preprocessor.transform(X_val)
    X_test  = preprocessor.transform(X_test)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val   = le.transform(y_val)
    y_test  = le.transform(y_test)

    X_train = torch.FloatTensor(X_train)
    X_val   = torch.FloatTensor(X_val)
    X_test  = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_val   = torch.FloatTensor(y_val)
    y_test  = torch.LongTensor(y_test)

    # feature indices for perturbation mask
    feature_indices = [numerical_cols.index(f)
                       for f in FEATURES_TO_PERTURB if f in numerical_cols]

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            feature_indices,
            X_train.min().item(), X_train.max().item())


# ─────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────
class CNNModel(nn.Module):
    def __init__(self, num_features, num_classes=1):
        super().__init__()
        layers = []
        in_channels = 1
        for _ in range(10):
            layers.append(nn.Conv1d(in_channels, 108, kernel_size=5, padding=2))
            layers.append(nn.BatchNorm1d(108))
            layers.append(nn.ReLU())
            in_channels = 108
        self.cnn_backbone = nn.Sequential(*layers)
        self.global_pool  = nn.AdaptiveAvgPool1d(1)
        self.classifier   = nn.Linear(108, num_classes)

    def forward(self, x, return_features=False):
        x        = x.unsqueeze(1)
        x        = self.cnn_backbone(x)
        features = self.global_pool(x).squeeze(-1)
        logit    = self.classifier(features)
        if return_features:
            return logit, features
        return logit


class LSTMModel(nn.Module):
    def __init__(self, num_features, hidden_size=128, num_layers=2, num_classes=1):
        super().__init__()
        self.lstm       = nn.LSTM(input_size=1, hidden_size=hidden_size,
                                  num_layers=num_layers, batch_first=True,
                                  dropout=0.2)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_features=False):
        # x: (B, features, 1) 
        x, _     = self.lstm(x.unsqueeze(-1))
        features = x[:, -1, :]        
        logit    = self.classifier(features)
        if return_features:
            return logit, features
        return logit


class CLSTMModel(nn.Module):
    """CNN feature extractor followed by LSTM."""
    def __init__(self, num_features, hidden_size=128, num_classes=1):
        super().__init__()
        # CNN part
        cnn_layers = []
        in_channels = 1
        for _ in range(5):
            cnn_layers.append(nn.Conv1d(in_channels, 64, kernel_size=5, padding=2))
            cnn_layers.append(nn.BatchNorm1d(64))
            cnn_layers.append(nn.ReLU())
            in_channels = 64
        self.cnn_backbone = nn.Sequential(*cnn_layers)

        # LSTM part — takes CNN output (B, 64, L) as (B, L, 64)
        self.lstm       = nn.LSTM(input_size=64, hidden_size=hidden_size,
                                  num_layers=2, batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_features=False):
        x        = x.unsqueeze(1)                   # (B, 1, features)
        x        = self.cnn_backbone(x)             # (B, 64, features)
        x        = x.permute(0, 2, 1)              # (B, features, 64)
        x, _     = self.lstm(x)
        features = x[:, -1, :]                      # (B, hidden)
        logit    = self.classifier(features)
        if return_features:
            return logit, features
        return logit


MODEL_REGISTRY = {
    'cnn':   CNNModel,
    'lstm':  LSTMModel,
    'clstm': CLSTMModel,
}

# ─────────────────────────────────────────
# Wrapper for foolbox
# ─────────────────────────────────────────
class BinaryWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logit = self.model(x)
        return torch.cat([torch.zeros_like(logit), logit], dim=1)

# ─────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────
def train_baseline(model, train_loader, val_loader, folder):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2_LAMBDA)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Baseline] Epoch {epoch+1}/{EPOCHS}")

        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(-1)
            loss    = criterion(outputs, labels.float())
            l1_reg  = sum(p.abs().sum() for p in model.parameters())
            loss    = loss + L1_LAMBDA * l1_reg
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                preds = (torch.sigmoid(model(inputs).squeeze(-1)) > 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds) * 100
        val_f1  = f1_score(all_labels, all_preds)
        print(f"  Val Acc: {val_acc:.2f}%  F1: {val_f1:.4f}")

    torch.save(model.state_dict(), os.path.join(folder, 'baseline.pth'))
    print(f"Baseline saved → {folder}/baseline.pth")
    return model


def evaluate_clean(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            preds = (torch.sigmoid(model(inputs).squeeze(-1)) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "accuracy":  accuracy_score(all_labels, all_preds) * 100,
        "precision": precision_score(all_labels, all_preds),
        "recall":    recall_score(all_labels, all_preds),
        "f1_score":  f1_score(all_labels, all_preds),
        "report":    classification_report(all_labels, all_preds,
                                           target_names=['Attack', 'Benign']),
    }


def evaluate_under_attack(model, test_loader, attack, epsilon, X_test):
    wrapped = BinaryWrapper(model).to(DEVICE)
    wrapped.train()
    fmodel  = fb.PyTorchModel(wrapped, bounds=(X_test.min().item(), X_test.max().item()))

    all_preds, all_labels = [], []

    for inputs, labels in tqdm(test_loader, desc=f"  Attack ε={epsilon}"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        is_malicious   = (labels == 0)
        adv_inputs     = inputs.clone()

        if is_malicious.sum() > 0:
            X_mal = inputs[is_malicious]
            y_mal = labels[is_malicious]
            with torch.enable_grad():
                _, adv_mal, _ = attack(fmodel, X_mal, y_mal.long(), epsilons=epsilon)
            adv_mal = torch.clamp(adv_mal, X_test.min().item(), X_test.max().item())
            adv_inputs = inputs.clone()
            adv_inputs[is_malicious] = adv_mal

        wrapped.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(model(adv_inputs).squeeze(-1)) > 0.5).long()
        wrapped.train()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    model.eval()

    return {
        "accuracy":  accuracy_score(all_labels, all_preds) * 100,
        "precision": precision_score(all_labels, all_preds),
        "recall":    recall_score(all_labels, all_preds),
        "f1_score":  f1_score(all_labels, all_preds),
        "report":    classification_report(all_labels, all_preds,
                                           target_names=['Attack', 'Benign']),
    }


# ─────────────────────────────────────────
# Teacher hardening
# ─────────────────────────────────────────
def harden_teacher(model_cls, num_features, baseline_path,
                   train_clean_loader, train_adv_loader,
                   val_loader, attack_obj, attack_name,
                   feature_indices, x_min, x_max, folder):

    model = model_cls(num_features).to(DEVICE)
    model.load_state_dict(torch.load(baseline_path, map_location=DEVICE))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2_LAMBDA)

    wrapped = BinaryWrapper(model).to(DEVICE)
    wrapped.train()
    fmodel  = fb.PyTorchModel(wrapped, bounds=(x_min, x_max))

    n_features        = next(iter(train_clean_loader))[0].shape[1]
    perturbation_mask = torch.zeros(n_features, device=DEVICE)
    perturbation_mask[feature_indices] = 1.0

    for epoch in range(EPOCHS):
        running_loss = 0.0
        pbar = tqdm(zip(train_clean_loader, train_adv_loader),
                    desc=f"[{attack_name} Teacher] Epoch {epoch+1}/{EPOCHS}",
                    total=min(len(train_clean_loader), len(train_adv_loader)))

        for (clean_inputs, clean_labels), (adv_inputs, adv_labels) in pbar:
            clean_inputs = clean_inputs.to(DEVICE)
            clean_labels = clean_labels.to(DEVICE)
            adv_inputs   = adv_inputs.to(DEVICE)
            adv_labels   = adv_labels.to(DEVICE)

            wrapped.train()
            is_malicious        = (adv_labels == 0)
            adv_inputs_perturbed = adv_inputs.clone()

            if is_malicious.any():
                X_mal   = adv_inputs[is_malicious]
                y_mal   = adv_labels[is_malicious].long()
                epsilon = random.uniform(0.01, 0.1)

                with torch.enable_grad():
                    _, adv_mal, _ = attack_obj(fmodel, X_mal, y_mal, epsilons=epsilon)

                adv_mal = X_mal + ((adv_mal - X_mal) * perturbation_mask)
                adv_mal = torch.clamp(adv_mal, x_min, x_max)

                nonzero_mask     = X_mal.abs() > 1e-2
                per_feature_mape = torch.abs((adv_mal - X_mal) / (X_mal.abs() + 1e-8)) * 100.0
                mape_ok          = ((per_feature_mape <= MAPE_THRESHOLD) | ~nonzero_mask).all(dim=1)
                adv_mal[~mape_ok] = X_mal[~mape_ok]

                adv_inputs_perturbed[is_malicious] = adv_mal

            mixed_inputs = torch.cat([clean_inputs, adv_inputs_perturbed], dim=0)
            mixed_labels = torch.cat([clean_labels, adv_labels], dim=0)

            optimizer.zero_grad()
            outputs = model(mixed_inputs).squeeze(-1)
            loss    = criterion(outputs, mixed_labels.float())
            l1_reg  = sum(p.abs().sum() for p in model.parameters())
            loss    = loss + L1_LAMBDA * l1_reg
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # validation
        wrapped.eval()
        all_preds, all_labels_list = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                preds = (torch.sigmoid(model(inputs).squeeze(-1)) > 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels_list, all_preds) * 100
        val_f1  = f1_score(all_labels_list, all_preds)
        print(f"  Val Acc: {val_acc:.2f}%  F1: {val_f1:.4f}")

        wrapped.train()

    save_path = os.path.join(folder, f'{attack_name.lower()}_teacher.pth')
    torch.save(model.state_dict(), save_path)
    print(f"{attack_name} teacher saved → {save_path}")
    return model


# ─────────────────────────────────────────
# Full pipeline per model
# ─────────────────────────────────────────
def run_pipeline_for_model(model_name, model_cls, num_features,
                           X_train, X_val, X_test,
                           y_train, y_val, y_test,
                           feature_indices, x_min, x_max):

    folder = model_name
    make_dir(folder)
    print(f"\n{'='*60}")
    print(f"  Running pipeline for: {model_name.upper()}")
    print(f"{'='*60}")

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),
                              batch_size=BATCH_SIZE, shuffle=False)

    # ── Baseline ──────────────────────────────────────────────────
    print("\n[1/3] Training baseline model...")
    baseline = model_cls(num_features).to(DEVICE)
    baseline = train_baseline(baseline, train_loader, val_loader, folder)

    print("\n  Evaluating baseline on clean test data...")
    baseline_clean_metrics = evaluate_clean(baseline, test_loader)
    save_json(baseline_clean_metrics,
              os.path.join(folder, 'baseline_clean_metrics.json'))
    print(f"  Accuracy: {baseline_clean_metrics['accuracy']:.2f}%")
    print(baseline_clean_metrics['report'])

    print("\n  Evaluating baseline under attacks...")
    attacks = {
        'fgsm':  FGSM(),
        'ifgsm': LinfBasicIterativeAttack(abs_stepsize=0.01, steps=10),
        'pgd':   PGD(rel_stepsize=0.02, steps=10, random_start=True),
    }
    baseline_attack_metrics = {}
    for atk_name, atk_obj in attacks.items():
        baseline_attack_metrics[atk_name] = {}
        for eps in EVAL_EPSILONS:
            m = evaluate_under_attack(baseline, test_loader, atk_obj, eps, X_test)
            baseline_attack_metrics[atk_name][str(eps)] = m
            print(f"  {atk_name.upper()} ε={eps} | Attack Recall: {m['recall']:.4f}")

    save_json(baseline_attack_metrics,
              os.path.join(folder, 'baseline_attack_metrics.json'))

    # ── Teacher hardening ──────────────────────────────────────────
    print("\n[2/3] Hardening teacher models...")

    # Split train into clean and adversarial pools
    indices   = torch.randperm(len(X_train))
    split     = len(X_train) // 2
    clean_idx = indices[:split]
    adv_idx   = indices[split:]

    X_clean, y_clean       = X_train[clean_idx], y_train[clean_idx]
    X_adv_pool, y_adv_pool = X_train[adv_idx],   y_train[adv_idx]

    train_clean_loader = DataLoader(TensorDataset(X_clean, y_clean),
                                    batch_size=BATCH_SIZE, shuffle=True)
    train_adv_loader   = DataLoader(TensorDataset(X_adv_pool, y_adv_pool),
                                    batch_size=BATCH_SIZE, shuffle=True)

    baseline_path = os.path.join(folder, 'baseline.pth')

    teacher_configs = [
        (FGSM(),                                               'FGSM'),
        (LinfBasicIterativeAttack(abs_stepsize=0.01, steps=10), 'IFGSM'),
        (PGD(rel_stepsize=0.02, steps=10, random_start=True),  'PGD'),
    ]

    teachers = {}
    for atk_obj, atk_name in teacher_configs:
        print(f"\n  Hardening {atk_name} teacher...")
        teacher = harden_teacher(
            model_cls, num_features, baseline_path,
            train_clean_loader, train_adv_loader,
            val_loader, atk_obj, atk_name,
            feature_indices, x_min, x_max, folder
        )
        teachers[atk_name.lower()] = teacher

    # ── Teacher evaluation ─────────────────────────────────────────
    print("\n[3/3] Evaluating teacher models...")
    teacher_metrics = {}

    for t_name, teacher in teachers.items():
        print(f"\n  Teacher: {t_name.upper()}")
        teacher_metrics[t_name] = {}

        # clean
        clean_m = evaluate_clean(teacher, test_loader)
        teacher_metrics[t_name]['clean'] = clean_m
        print(f"  Clean Accuracy: {clean_m['accuracy']:.2f}%")
        print(clean_m['report'])

        # under each attack
        teacher_metrics[t_name]['attacks'] = {}
        for atk_name, atk_obj in attacks.items():
            teacher_metrics[t_name]['attacks'][atk_name] = {}
            for eps in EVAL_EPSILONS:
                m = evaluate_under_attack(teacher, test_loader, atk_obj, eps, X_test)
                teacher_metrics[t_name]['attacks'][atk_name][str(eps)] = m
                print(f"  {atk_name.upper()} ε={eps} | Attack Recall: {m['recall']:.4f}")

    save_json(teacher_metrics,
              os.path.join(folder, 'teacher_metrics.json'))

    print(f"\n  All metrics saved to ./{folder}/")
    print(f"  Model paths:")
    print(f"    Baseline : {folder}/baseline.pth")
    for t_name in teachers:
        print(f"    {t_name.upper()} Teacher: {folder}/{t_name}_teacher.pth")

    return teachers


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == '__main__':

    print("Loading and preprocessing dataset...")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_indices,
     x_min, x_max) = load_and_preprocess("merged_IDS2018.csv")

    num_features = X_train.shape[1]
    print(f"Features: {num_features} | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Run pipeline for each model
    all_results = {}
    for model_name, model_cls in MODEL_REGISTRY.items():
        teachers = run_pipeline_for_model(
            model_name, model_cls, num_features,
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            feature_indices, x_min, x_max,
        )
        all_results[model_name] = teachers

    print("\n" + "="*60)
    print("  Pipeline complete for all models.")
    print("  Folder structure:")
    for model_name in MODEL_REGISTRY:
        print(f"    {model_name}/")
        print(f"      baseline.pth")
        print(f"      baseline_clean_metrics.json")
        print(f"      baseline_attack_metrics.json")
        print(f"      fgsm_teacher.pth")
        print(f"      ifgsm_teacher.pth")
        print(f"      pgd_teacher.pth")
        print(f"      teacher_metrics.json")
    print("="*60)
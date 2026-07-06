# ──────────────────────────────────────────────────────────────────────────
#  Adversarially-Robust NIDS — multi-architecture pipeline & defense comparison
#
#  For EACH architecture (CNN, LSTM, CLSTM) trains everything needed to compare
#  two adversarial-defense strategies, then evaluates them:
#
#     1. Baseline            — standard training, no defense (reference)
#     2. Union AT            — single model, random attack+ε each batch
#     3. MTKD-ADR student    — distilled from 3 attack-specialised teachers
#
#  All models are evaluated CLEAN and under FGSM / I-FGSM / PGD at several ε,
#  using a WHITE-BOX ADAPTIVE attack (crafted against the model under test).
#  Headline robustness metric: Attack-class recall (malicious flows still caught).
#
#  Output layout (one folder per architecture):
#     comparison/<arch>/baseline.pth, union_at.pth, mtkd_student.pth,
#                       teacher_{fgsm,ifgsm,pgd}.pth, comparison_results.json
#     comparison/comparison_all_archs.json   (combined, keyed by architecture)
#
#  Any stage whose .pth already exists is REUSED (skipped), so an interrupted
#  run resumes, and a finished architecture (e.g. CNN) is never retrained.
#
#  Usage:
#     python pipeline.py                       # all 3 archs, train missing + evaluate
#     python pipeline.py --arch lstm,clstm     # only these architectures
#     python pipeline.py --eval-only           # evaluate saved checkpoints only
#     python pipeline.py --epochs 5
# ──────────────────────────────────────────────────────────────────────────
import os
import json
import random
import argparse
import contextlib

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
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────────
SEED          = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR       = "comparison"
BATCH_SIZE    = 1024
EPOCHS        = 5
LR            = 1e-3
L2_LAMBDA     = 1e-5
L1_LAMBDA     = 1e-6
TEMPERATURE   = 4.0
ALPHA         = 0.4           # KD-loss weight        (MTKD)
BETA          = 0.4           # feature-loss weight   (MTKD)  -> CE weight = 1-α-β
MAPE_THRESHOLD = 20.0
BOUNDS        = (0.0, 1.0)
TRAIN_EPSILON_RANGE = [0.03, 0.07, 0.1]
EVAL_EPSILONS       = [0.03, 0.07, 0.1]

# Set per-architecture in the main loop: recurrent nets need cuDNN disabled
# during attack generation (cuDNN RNN backward is forbidden in eval mode).
_DISABLE_CUDNN = False

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

ATTACK_LABEL = 0   # LabelEncoder: Attack->0, Benign->1
BENIGN_LABEL = 1


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_attacks():
    return {
        'fgsm':  FGSM(),
        'ifgsm': LinfBasicIterativeAttack(abs_stepsize=0.01, steps=20),
        'pgd':   PGD(rel_stepsize=0.02, steps=40, random_start=True),
    }


# ──────────────────────────────────────────────────────────────────────────
#  Data loading & preprocessing
# ──────────────────────────────────────────────────────────────────────────
def load_and_preprocess(csv_path):
    print("Loading and preprocessing data...")
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
        X, y_binary, test_size=0.2, random_state=SEED, stratify=y_binary)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=SEED, stratify=y_test)

    ros = RandomOverSampler(sampling_strategy='auto', random_state=SEED)
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
    print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X_train = torch.FloatTensor(X_train); X_val = torch.FloatTensor(X_val); X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train); y_val = torch.FloatTensor(y_val); y_test = torch.LongTensor(y_test)

    feature_indices = [numerical_cols.index(f)
                       for f in FEATURES_TO_PERTURB if f in numerical_cols]

    print(f"Features: {X_train.shape[1]} | Train: {len(X_train)} | "
          f"Val: {len(X_val)} | Test: {len(X_test)}")
    return (X_train, X_val, X_test, y_train, y_val, y_test, feature_indices)


# ──────────────────────────────────────────────────────────────────────────
#  Models   (all share signature  __init__(num_features, num_classes=1)
#            and  forward(x, return_features=False) -> logit  or  (logit, feats))
# ──────────────────────────────────────────────────────────────────────────
class CNNModel(nn.Module):
    """1-D CNN binary detector (feature-count agnostic via global avg pool)."""
    def __init__(self, num_features=None, num_classes=1):
        super().__init__()
        layers = []
        in_channels = 1
        for _ in range(10):
            layers += [nn.Conv1d(in_channels, 108, kernel_size=5, padding=2),
                       nn.BatchNorm1d(108), nn.ReLU()]
            in_channels = 108
        self.cnn_backbone = nn.Sequential(*layers)
        self.global_pool  = nn.AdaptiveAvgPool1d(1)
        self.classifier   = nn.Linear(108, num_classes)

    def forward(self, x, return_features=False):
        x        = x.unsqueeze(1)
        x        = self.cnn_backbone(x)
        features = self.global_pool(x).squeeze(-1)
        logit    = self.classifier(features)
        return (logit, features) if return_features else logit


class LSTMModel(nn.Module):
    def __init__(self, num_features=None, hidden_size=128, num_layers=2, num_classes=1):
        super().__init__()
        self.lstm       = nn.LSTM(input_size=1, hidden_size=hidden_size,
                                  num_layers=num_layers, batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_features=False):
        x, _     = self.lstm(x.unsqueeze(-1))   # (B, features, 1) -> (B, features, H)
        features = x[:, -1, :]
        logit    = self.classifier(features)
        return (logit, features) if return_features else logit


class CLSTMModel(nn.Module):
    """CNN feature extractor followed by LSTM."""
    def __init__(self, num_features=None, hidden_size=128, num_classes=1):
        super().__init__()
        cnn_layers = []
        in_channels = 1
        for _ in range(5):
            cnn_layers += [nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
                           nn.BatchNorm1d(64), nn.ReLU()]
            in_channels = 64
        self.cnn_backbone = nn.Sequential(*cnn_layers)
        self.lstm       = nn.LSTM(input_size=64, hidden_size=hidden_size,
                                  num_layers=2, batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_features=False):
        x        = x.unsqueeze(1)               # (B, 1, F)
        x        = self.cnn_backbone(x)         # (B, 64, F)
        x        = x.permute(0, 2, 1)           # (B, F, 64)
        x, _     = self.lstm(x)
        features = x[:, -1, :]
        logit    = self.classifier(features)
        return (logit, features) if return_features else logit


MODEL_REGISTRY = {'cnn': CNNModel, 'lstm': LSTMModel, 'clstm': CLSTMModel}
RECURRENT      = {'lstm', 'clstm'}


class BinaryWrapper(nn.Module):
    """Adapts the single-logit output to 2-class logits for foolbox."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logit = self.model(x)
        return torch.cat([torch.zeros_like(logit), logit], dim=1)


def make_fmodel(model):
    wrapped = BinaryWrapper(model).to(DEVICE)
    wrapped.eval()
    return fb.PyTorchModel(wrapped, bounds=BOUNDS)


# ──────────────────────────────────────────────────────────────────────────
#  Adversarial-example generation (shared by training & evaluation)
# ──────────────────────────────────────────────────────────────────────────
def craft_adv(attack, fmodel, inputs, labels, epsilon, mask=None):
    """White-box attack on malicious (label==ATTACK_LABEL) rows only.

    Returns a full-batch tensor: attack rows perturbed, benign rows untouched.
    `mask` (optional) restricts the perturbation to selected features.
    cuDNN is disabled here for recurrent archs (RNN backward fails in eval mode).
    """
    adv = inputs.clone()
    is_mal = (labels == ATTACK_LABEL)
    if is_mal.sum() == 0:
        return adv

    x_mal = inputs[is_mal]
    y_mal = labels[is_mal].long()
    cudnn_cm = (torch.backends.cudnn.flags(enabled=False)
                if _DISABLE_CUDNN else contextlib.nullcontext())
    with cudnn_cm, torch.enable_grad():
        _, adv_mal, _ = attack(fmodel, x_mal, y_mal, epsilons=epsilon)

    if mask is not None:
        adv_mal = x_mal + (adv_mal - x_mal) * mask
    adv_mal = torch.clamp(adv_mal, BOUNDS[0], BOUNDS[1])
    adv[is_mal] = adv_mal
    return adv


# ──────────────────────────────────────────────────────────────────────────
#  Stage 1 — Baseline
# ──────────────────────────────────────────────────────────────────────────
def train_baseline(model_cls, num_features, train_loader, val_loader):
    print("\n[Stage 1] Training baseline (undefended)...")
    model     = model_cls(num_features).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2_LAMBDA)
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Baseline] Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(inputs).squeeze(-1)
            loss = criterion(out, labels.float())
            loss = loss + L1_LAMBDA * sum(p.abs().sum() for p in model.parameters())
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        _report_val(model, val_loader, epoch)
    return model


# ──────────────────────────────────────────────────────────────────────────
#  Stage 2 — Teacher hardening (realistic-feature mask + MAPE constraint)
# ──────────────────────────────────────────────────────────────────────────
def harden_teacher(model_cls, num_features, baseline_state, attack, attack_name,
                   clean_loader, adv_loader, val_loader, feature_indices, n_features):
    print(f"\n[Stage 2] Hardening {attack_name.upper()} teacher...")
    model = model_cls(num_features).to(DEVICE)
    model.load_state_dict(baseline_state)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2_LAMBDA)
    fmodel    = make_fmodel(model)

    mask = torch.zeros(n_features, device=DEVICE)
    mask[feature_indices] = 1.0

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(zip(clean_loader, adv_loader),
                    desc=f"[{attack_name.upper()} Teacher] Epoch {epoch+1}/{EPOCHS}",
                    total=min(len(clean_loader), len(adv_loader)))
        for (clean_x, clean_y), (adv_x, adv_y) in pbar:
            clean_x, clean_y = clean_x.to(DEVICE), clean_y.to(DEVICE)
            adv_x,   adv_y   = adv_x.to(DEVICE),   adv_y.to(DEVICE)

            epsilon  = random.uniform(0.01, 0.1)
            adv_pert = craft_adv(attack, fmodel, adv_x, adv_y, epsilon, mask=mask)

            # drop perturbations exceeding the MAPE budget (keep them realistic)
            is_mal = (adv_y == ATTACK_LABEL)
            if is_mal.any():
                x_mal = adv_x[is_mal]
                a_mal = adv_pert[is_mal]
                nonzero = x_mal.abs() > 1e-2
                mape    = torch.abs((a_mal - x_mal) / (x_mal.abs() + 1e-8)) * 100.0
                ok      = ((mape <= MAPE_THRESHOLD) | ~nonzero).all(dim=1)
                a_mal[~ok] = x_mal[~ok]
                adv_pert[is_mal] = a_mal

            mixed_x = torch.cat([clean_x, adv_pert], dim=0)
            mixed_y = torch.cat([clean_y, adv_y],    dim=0)

            optimizer.zero_grad()
            out  = model(mixed_x).squeeze(-1)
            loss = criterion(out, mixed_y.float())
            loss = loss + L1_LAMBDA * sum(p.abs().sum() for p in model.parameters())
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        _report_val(model, val_loader, epoch)
    return model


# ──────────────────────────────────────────────────────────────────────────
#  Stage 3 — Union adversarial training
# ──────────────────────────────────────────────────────────────────────────
def train_union(model_cls, num_features, baseline_state, clean_loader, adv_loader, val_loader):
    print("\n[Stage 3] Training Union adversarial model...")
    model     = model_cls(num_features).to(DEVICE)
    model.load_state_dict(baseline_state)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2_LAMBDA)
    attacks   = list(make_attacks().values())

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(zip(clean_loader, adv_loader),
                    desc=f"[Union] Epoch {epoch+1}/{EPOCHS}",
                    total=min(len(clean_loader), len(adv_loader)))
        for (clean_x, clean_y), (adv_x, adv_y) in pbar:
            clean_x, clean_y = clean_x.to(DEVICE), clean_y.to(DEVICE)
            adv_x,   adv_y   = adv_x.to(DEVICE),   adv_y.to(DEVICE)

            attack   = random.choice(attacks)
            epsilon  = random.choice(TRAIN_EPSILON_RANGE)
            fmodel   = make_fmodel(model)
            adv_pert = craft_adv(attack, fmodel, adv_x, adv_y, epsilon)

            mixed_x = torch.cat([clean_x, adv_pert], dim=0)
            mixed_y = torch.cat([clean_y, adv_y],    dim=0)

            model.train()
            optimizer.zero_grad()
            out  = model(mixed_x).squeeze(-1)
            loss = criterion(out, mixed_y.float())
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}",
                              'atk': attack.__class__.__name__, 'eps': f"{epsilon:.2f}"})
        _report_val(model, val_loader, epoch)
    return model


# ──────────────────────────────────────────────────────────────────────────
#  Stage 4 — MTKD-ADR student
# ──────────────────────────────────────────────────────────────────────────
def compute_adaptive_weights(student_logits, teacher_logits_list):
    sims = [F.cosine_similarity(student_logits, t, dim=0).mean()
            for t in teacher_logits_list]
    weights = [1.0 + s for s in sims]
    total = sum(weights)
    return [w / total for w in weights]


def distillation_loss(student_logits, teacher_logits_list, weights, T=TEMPERATURE):
    s_soft = torch.sigmoid(student_logits / T)
    s_probs = torch.cat([1 - s_soft, s_soft], dim=1)
    s_log   = torch.log(s_probs.clamp(min=1e-8))
    weighted_t = torch.zeros_like(s_soft)
    for t_logits, w in zip(teacher_logits_list, weights):
        weighted_t += w * torch.sigmoid(t_logits / T)
    t_probs = torch.cat([1 - weighted_t, weighted_t], dim=1)
    return F.kl_div(s_log, t_probs.clamp(min=1e-8), reduction='batchmean') * (T ** 2)


def feature_distillation_loss(student_features, teacher_features_list, weights):
    loss = torch.zeros((), device=student_features.device)
    for t_feat, w in zip(teacher_features_list, weights):
        loss = loss + w * F.mse_loss(student_features, t_feat.detach())
    return loss


def train_mtkd_student(model_cls, num_features, teachers, train_loader, val_loader):
    print("\n[Stage 4] Training MTKD-ADR student...")
    student   = model_cls(num_features).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(student.parameters(), lr=LR, weight_decay=L2_LAMBDA)

    teacher_list = list(teachers.values())
    for t in teacher_list:
        t.eval()
    fmodels = [make_fmodel(t) for t in teacher_list]
    attacks = list(make_attacks().values())

    for epoch in range(EPOCHS):
        student.train()
        pbar = tqdm(train_loader, desc=f"[MTKD] Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            epsilon = random.choice(TRAIN_EPSILON_RANGE)

            adv_batches = [craft_adv(atk, fm, inputs, labels, epsilon)
                           for atk, fm in zip(attacks, fmodels)]
            all_x = torch.cat([inputs] + adv_batches, dim=0)
            all_y = torch.cat([labels] * 4, dim=0)

            optimizer.zero_grad()
            s_logits, s_feats = student(all_x, return_features=True)
            with torch.no_grad():
                t_logits_list, t_feats_list = [], []
                for t in teacher_list:
                    tl, tf = t(all_x, return_features=True)
                    t_logits_list.append(tl); t_feats_list.append(tf)

            weights = compute_adaptive_weights(s_logits.detach(), t_logits_list)
            ce   = criterion(s_logits.squeeze(-1), all_y.float())
            kd   = distillation_loss(s_logits, t_logits_list, weights)
            feat = feature_distillation_loss(s_feats, t_feats_list, weights)
            loss = (1 - ALPHA - BETA) * ce + ALPHA * kd + BETA * feat
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'ce': f"{ce.item():.3f}",
                              'kd': f"{kd.item():.3f}", 'feat': f"{feat.item():.3f}"})
        _report_val(student, val_loader, epoch)
    return student


# ──────────────────────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────────────────────
def _metrics(y_true, y_pred):
    return {
        "accuracy":      accuracy_score(y_true, y_pred),
        "attack_recall": recall_score(y_true, y_pred, pos_label=ATTACK_LABEL, zero_division=0),
        "benign_recall": recall_score(y_true, y_pred, pos_label=BENIGN_LABEL, zero_division=0),
        "attack_precision": precision_score(y_true, y_pred, pos_label=ATTACK_LABEL, zero_division=0),
        "macro_f1":      f1_score(y_true, y_pred, average='macro', zero_division=0),
        "report":        classification_report(y_true, y_pred, target_names=['Attack', 'Benign'],
                                               output_dict=True, zero_division=0),
    }


@torch.no_grad()
def evaluate_clean(model, loader):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        p = (torch.sigmoid(model(x).squeeze(-1)) > 0.5).long().cpu().numpy()
        preds.extend(p); labels.extend(y.numpy())
    return _metrics(labels, preds)


def evaluate_under_attack(model, loader, attack, epsilon):
    model.eval()
    fmodel = make_fmodel(model)
    preds, labels = [], []
    for x, y in tqdm(loader, desc=f"  {attack.__class__.__name__} ε={epsilon}", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        adv  = craft_adv(attack, fmodel, x, y, epsilon)
        with torch.no_grad():
            p = (torch.sigmoid(model(adv).squeeze(-1)) > 0.5).long().cpu().numpy()
        preds.extend(p); labels.extend(y.cpu().numpy())
    return _metrics(labels, preds)


def _report_val(model, val_loader, epoch):
    m = evaluate_clean(model, val_loader)
    print(f"  Epoch {epoch+1}: Val Acc {m['accuracy']*100:.2f}% | "
          f"Macro-F1 {m['macro_f1']:.4f} | Attack-Recall {m['attack_recall']:.4f}")


def evaluate_all(models, test_loader):
    print("\n" + "=" * 70)
    print("  EVALUATION — clean + white-box adaptive attacks")
    print("=" * 70)
    results = {}
    for name, model in models.items():
        print(f"\n>>> Model: {name}")
        results[name] = {'clean': evaluate_clean(model, test_loader), 'attacks': {}}
        print(f"    clean | Acc {results[name]['clean']['accuracy']*100:.2f}% "
              f"| Attack-Recall {results[name]['clean']['attack_recall']:.4f}")
        for atk_name, atk in make_attacks().items():
            results[name]['attacks'][atk_name] = {}
            for eps in EVAL_EPSILONS:
                m = evaluate_under_attack(model, test_loader, atk, eps)
                results[name]['attacks'][atk_name][str(eps)] = m
                print(f"    {atk_name:<5} ε={eps:<4} | Acc {m['accuracy']*100:5.2f}% "
                      f"| Attack-Recall {m['attack_recall']:.4f}")
    return results


def print_comparison_table(results, arch):
    models = list(results.keys())
    print("\n" + "=" * 72)
    print(f"  [{arch.upper()}] Attack-class recall (↑ better; malicious flows still caught)")
    print("=" * 72)
    header = f"{'condition':<16}" + "".join(f"{m:>16}" for m in models)
    print(header); print("-" * len(header))
    print(f"{'clean':<16}" + "".join(f"{results[m]['clean']['attack_recall']:>16.4f}" for m in models))
    for atk in ['fgsm', 'ifgsm', 'pgd']:
        for eps in EVAL_EPSILONS:
            row = f"{atk+' ε='+str(eps):<16}"
            row += "".join(f"{results[m]['attacks'][atk][str(eps)]['attack_recall']:>16.4f}"
                           for m in models)
            print(row)
    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────────
#  Per-architecture driver
# ──────────────────────────────────────────────────────────────────────────
def load_ckpt(model_cls, num_features, path):
    m = model_cls(num_features).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


def run_architecture(arch, data, eval_only):
    global _DISABLE_CUDNN
    _DISABLE_CUDNN = arch in RECURRENT
    model_cls = MODEL_REGISTRY[arch]
    (X_train, X_val, X_test, y_train, y_val, y_test, feature_indices) = data
    n_features = X_train.shape[1]

    folder = os.path.join(OUT_DIR, arch)
    os.makedirs(folder, exist_ok=True)
    paths = {k: os.path.join(folder, f'{k}.pth') for k in
             ['baseline', 'teacher_fgsm', 'teacher_ifgsm', 'teacher_pgd', 'union_at', 'mtkd_student']}

    print("\n" + "#" * 72)
    print(f"#  ARCHITECTURE: {arch.upper()}   (recurrent={arch in RECURRENT})")
    print("#" * 72)

    val_loader  = DataLoader(TensorDataset(X_val,  y_val),  batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    if eval_only:
        models = {'baseline': load_ckpt(model_cls, n_features, paths['baseline']),
                  'union_at': load_ckpt(model_cls, n_features, paths['union_at']),
                  'mtkd_adr': load_ckpt(model_cls, n_features, paths['mtkd_student'])}
    else:
        train_loader = DataLoader(TensorDataset(X_train, y_train),
                                  batch_size=BATCH_SIZE, shuffle=True)

        # Stage 1 — baseline (reuse if present)
        if os.path.exists(paths['baseline']):
            print(f"[skip] baseline exists → {paths['baseline']}")
            baseline = load_ckpt(model_cls, n_features, paths['baseline'])
        else:
            baseline = train_baseline(model_cls, n_features, train_loader, val_loader)
            torch.save(baseline.state_dict(), paths['baseline'])
        baseline_state = baseline.state_dict()

        # clean / adversarial halves for the AT stages
        idx   = torch.randperm(len(X_train))
        split = len(X_train) // 2
        clean_loader = DataLoader(TensorDataset(X_train[idx[:split]], y_train[idx[:split]]),
                                  batch_size=BATCH_SIZE, shuffle=True)
        adv_loader   = DataLoader(TensorDataset(X_train[idx[split:]], y_train[idx[split:]]),
                                  batch_size=BATCH_SIZE, shuffle=True)

        # Stage 2 — teachers (reuse if present)
        teachers = {}
        for name, atk in make_attacks().items():
            tpath = paths[f'teacher_{name}']
            if os.path.exists(tpath):
                print(f"[skip] {name} teacher exists → {tpath}")
                teachers[name] = load_ckpt(model_cls, n_features, tpath)
            else:
                t = harden_teacher(model_cls, n_features, baseline_state, atk, name,
                                   clean_loader, adv_loader, val_loader, feature_indices, n_features)
                torch.save(t.state_dict(), tpath)
                teachers[name] = t

        # Stage 3 — Union AT (reuse if present)
        if os.path.exists(paths['union_at']):
            print(f"[skip] union exists → {paths['union_at']}")
            union = load_ckpt(model_cls, n_features, paths['union_at'])
        else:
            union = train_union(model_cls, n_features, baseline_state,
                                clean_loader, adv_loader, val_loader)
            torch.save(union.state_dict(), paths['union_at'])

        # Stage 4 — MTKD-ADR student (reuse if present)
        if os.path.exists(paths['mtkd_student']):
            print(f"[skip] mtkd student exists → {paths['mtkd_student']}")
            student = load_ckpt(model_cls, n_features, paths['mtkd_student'])
        else:
            student = train_mtkd_student(model_cls, n_features, teachers, train_loader, val_loader)
            torch.save(student.state_dict(), paths['mtkd_student'])

        models = {'baseline': baseline, 'union_at': union, 'mtkd_adr': student}

    results = evaluate_all(models, test_loader)
    with open(os.path.join(folder, 'comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print_comparison_table(results, arch)
    return results


# ──────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    global EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='merged_IDS2018.csv')
    ap.add_argument('--arch', default='cnn,lstm,clstm',
                    help='comma-separated subset of: cnn,lstm,clstm')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--eval-only', action='store_true',
                    help='skip training; evaluate saved checkpoints only')
    args = ap.parse_args()
    EPOCHS = args.epochs
    archs = [a.strip() for a in args.arch.split(',') if a.strip()]
    for a in archs:
        if a not in MODEL_REGISTRY:
            raise SystemExit(f"unknown arch '{a}' (choose from {list(MODEL_REGISTRY)})")

    set_seed()
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_and_preprocess(args.csv)

    all_results = {}
    for arch in archs:
        all_results[arch] = run_architecture(arch, data, args.eval_only)

    with open(os.path.join(OUT_DIR, 'comparison_all_archs.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results → {os.path.join(OUT_DIR, 'comparison_all_archs.json')}")


if __name__ == '__main__':
    main()

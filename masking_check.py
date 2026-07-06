# Gradient-masking diagnostic: escalate PGD ε on each saved checkpoint.
# If attack-recall stays high at large ε, the gradients are masked and the
# "robustness" is an artifact. A genuinely (bounded-)robust model must still
# collapse toward recall≈0 as ε→1.0 (the max possible Linf step in [0,1]).
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
import pipeline as P

P.set_seed()
SUBSET = 60000
EPS_SWEEP = [0.03, 0.1, 0.2, 0.3, 0.5, 1.0]

(X_train, X_val, X_test, y_train, y_val, y_test, _) = P.load_and_preprocess("merged_IDS2018.csv")

# fixed random subset of the test set (deterministic via set_seed)
idx = torch.randperm(len(X_test))[:SUBSET]
Xs, ys = X_test[idx], y_test[idx]
n_attack = int((ys == P.ATTACK_LABEL).sum())
print(f"\nDiagnostic subset: {len(Xs)} rows ({n_attack} attack / {len(Xs)-n_attack} benign)")
loader = DataLoader(TensorDataset(Xs, ys), batch_size=P.BATCH_SIZE, shuffle=False)

ckpts = {
    'baseline': 'comparison/baseline.pth',
    'union_at': 'comparison/union_at.pth',
    'mtkd_adr': 'comparison/mtkd_student.pth',
}
models = {}
for name, path in ckpts.items():
    m = P.DDoSDetector().to(P.DEVICE)
    m.load_state_dict(torch.load(path, map_location=P.DEVICE))
    m.eval()
    models[name] = m

pgd = P.make_attacks()['pgd']   # 40-step PGD
results = {name: {} for name in models}
for name, m in models.items():
    print(f"\n=== {name} — PGD ε sweep (attack-recall) ===")
    clean = P.evaluate_clean(m, loader)
    results[name]['clean'] = clean['attack_recall']
    print(f"  clean        recall={clean['attack_recall']:.4f}")
    for eps in EPS_SWEEP:
        r = P.evaluate_under_attack(m, loader, pgd, eps)['attack_recall']
        results[name][str(eps)] = r
        print(f"  pgd ε={eps:<4}   recall={r:.4f}")

# summary table
print("\n" + "=" * 64)
print("  GRADIENT-MASKING CHECK — PGD attack-recall vs ε  (↓ to 0 = healthy)")
print("=" * 64)
cols = list(models.keys())
print(f"{'ε':<10}" + "".join(f"{c:>16}" for c in cols))
print("-" * (10 + 16 * len(cols)))
print(f"{'clean':<10}" + "".join(f"{results[c]['clean']:>16.4f}" for c in cols))
for eps in EPS_SWEEP:
    print(f"{eps:<10}" + "".join(f"{results[c][str(eps)]:>16.4f}" for c in cols))
print("=" * 64)

with open('comparison/masking_check.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved → comparison/masking_check.json")

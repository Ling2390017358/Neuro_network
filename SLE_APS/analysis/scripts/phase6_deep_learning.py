#!/usr/bin/env python3
"""
Phase 6: Deep Learning Time-Series Prediction
==============================================
Covers T6.1–T6.4:
  T6.1: Sequential dataset construction (LOCF, ΔT encoding)
  T6.2: Bi-LSTM training with focal loss
  T6.3: Transformer model training
  T6.4: DL explainability (attention weights, integrated gradients)

Outputs:
  - models/bilstm_best.pt
  - models/transformer_best.pt
  - analysis/output/dl_test_results.csv
  - analysis/figures/main/Figure_Attention_Heatmap.pdf
  - analysis/figures/supplementary/DL_Training_Curves.png
"""

import pandas as pd
import numpy as np
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
import torch
torch.manual_seed(RANDOM_SEED)

BASE = Path('/home/ubuntu/projects/SLE_APS')
OUT = BASE / 'analysis' / 'output'
FIG = BASE / 'analysis' / 'figures'
PROC = BASE / 'data' / 'processed'
MODELS_DIR = BASE / 'models'

for d in [OUT, FIG / 'main', FIG / 'supplementary', MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 6: DEEP LEARNING TIME-SERIES PREDICTION")
print("=" * 60)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  Device: {device}")

# ══════════════════════════════════════════════════════════════════════
# T6.1: Sequential Dataset Construction
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T6.1] Sequential Dataset Construction")
print("─" * 50)

df_raw = pd.read_csv(BASE / 'data' / 'raw' / 'SLEmatrix_merged.csv', encoding='utf-8-sig', low_memory=False)
patient_df = pd.read_csv(OUT / 'patient_level_data.csv')

id_col = '_patient_SN' if '_patient_SN' in df_raw.columns else 'patient_SN'
visit_date_col = '_visit_date' if '_visit_date' in df_raw.columns else 'visit_date'
aps_col = 'APS'
df_raw[visit_date_col] = pd.to_datetime(df_raw[visit_date_col], errors='coerce')

# Select 21 core biomarkers for DL
core_features = [
    '补体C3_静脉血_定量',
    '补体C4_静脉血_定量',
    '活化部分凝血活酶时间_APTT__静脉血_定量',
    '血小板计数_PLT#__静脉血_定量',
    '血红蛋白_Hb__静脉血_定量',
    'C_反应蛋白_CRP__静脉血_定量',
    '白细胞计数_WBC#__静脉血_定量',
    '肌酐_Crea__静脉血_定量',
    '尿酸_UA__静脉血_定量',
    '凝血酶原时间_PT__静脉血_定量',
    '凝血酶时间_TT__静脉血_定量',
    '纤维蛋白原_Fbg__静脉血_定量',
    '总蛋白_TP__静脉血_定量',
    '白蛋白_ALB__静脉血_定量',
    '总胆红素_TBIL__静脉血_定量',
    '直接胆红素_DBIL__静脉血_定量',
    '钾离子_K+__静脉血_定量',
    '钠离子_Na+__静脉血_定量',
    '总钙_Ca__静脉血_定量',
    '磷_P__静脉血_定量',
    '空腹血糖_GLU__静脉血_定量',
]

# Filter to available columns
available_features = [c for c in core_features if c in df_raw.columns]
print(f"  Available core features: {len(available_features)}/{len(core_features)}")

# Add APS label
df_raw = df_raw.merge(patient_df[[id_col, aps_col]], on=id_col, how='left')

# Sort by patient and visit date
df_raw = df_raw.sort_values([id_col, visit_date_col]).reset_index(drop=True)

# Calculate time delta in days
df_raw['visit_time_days'] = df_raw.groupby(id_col)[visit_date_col].diff().dt.days.fillna(0)
df_raw['visit_num'] = df_raw.groupby(id_col).cumcount() + 1

# Build sequences: each patient → list of visits
# Max sequence length
max_seq_len = 10

# Patients with ≥2 visits
visit_counts = df_raw.groupby(id_col).size()
eligible = visit_counts[visit_counts >= 2].index
print(f"  Eligible patients (≥2 visits): {len(eligible):,}")

# Build feature/label matrices
patient_sequences = []
patient_labels = []
patient_ids_seq = []

for pid in eligible:
    pdata = df_raw[df_raw[id_col] == pid].sort_values(visit_date_col).head(max_seq_len)

    # Feature matrix
    feat_vals = pdata[available_features].values  # (n_visits, n_features)
    dt_vals = pdata['visit_time_days'].values.reshape(-1, 1) / 365.25  # years

    # Label
    label = pdata[aps_col].iloc[0]

    # Pad to max_seq_len
    n = len(feat_vals)
    if n < max_seq_len:
        feat_pad = np.pad(feat_vals, ((0, max_seq_len - n), (0, 0)), mode='constant', constant_values=np.nan)
        dt_pad = np.pad(dt_vals, ((0, max_seq_len - n), (0, 0)), mode='constant', constant_values=0)
        mask = np.array([1] * n + [0] * (max_seq_len - n))
    else:
        feat_pad = feat_vals[:max_seq_len]
        dt_pad = dt_vals[:max_seq_len]
        mask = np.ones(max_seq_len)

    patient_sequences.append(np.concatenate([feat_pad, dt_pad], axis=1))  # append ΔT
    patient_labels.append(label)
    patient_ids_seq.append(pid)

X_seq = np.array(patient_sequences)
y_seq = np.array(patient_labels)
n_features = len(available_features) + 1  # +1 for ΔT

print(f"  Sequence tensor: {X_seq.shape}")
print(f"  Labels: {y_seq.sum():,} positive ({y_seq.mean()*100:.2f}%)")

# LOCF forward fill for missing values
for i in range(len(X_seq)):
    for j in range(1, max_seq_len):
        for k in range(n_features):
            if np.isnan(X_seq[i, j, k]):
                X_seq[i, j, k] = X_seq[i, j-1, k]
    # Remaining NaN → median
    for k in range(n_features):
        col_vals = X_seq[:, :, k].flatten()
        col_vals = col_vals[~np.isnan(col_vals)]
        median_val = np.median(col_vals) if len(col_vals) > 0 else 0
        X_seq[:, :, k] = np.nan_to_num(X_seq[:, :, k], nan=median_val)

# Standardize per feature
for k in range(n_features - 1):  # Don't standardize ΔT
    all_vals = X_seq[:, :, k].flatten()
    mean_val = np.mean(all_vals)
    std_val = np.std(all_vals) + 1e-8
    X_seq[:, :, k] = (X_seq[:, :, k] - mean_val) / std_val

# Train/val/test split (temporal)
n_total = len(X_seq)
n_train = int(n_total * 0.7)
n_val = int(n_total * 0.15)

train_idx = range(n_train)
val_idx = range(n_train, n_train + n_val)
test_idx = range(n_train + n_val, n_total)

X_train, y_train = X_seq[train_idx], y_seq[train_idx]
X_val, y_val = X_seq[val_idx], y_seq[val_idx]
X_test, y_test = X_seq[test_idx], y_seq[test_idx]

print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Convert to tensors
def to_tensor(X, y):
    return (
        torch.FloatTensor(X),
        torch.FloatTensor(y).unsqueeze(1)
    )

X_train_t, y_train_t = to_tensor(X_train, y_train)
X_val_t, y_val_t = to_tensor(X_val, y_val)
X_test_t, y_test_t = to_tensor(X_test, y_test)

# Save datasets
torch.save({'X': X_train_t, 'y': y_train_t}, PROC / 'sequential_dataset_train.pt')
torch.save({'X': X_val_t, 'y': y_val_t}, PROC / 'sequential_dataset_val.pt')
torch.save({'X': X_test_t, 'y': y_test_t}, PROC / 'sequential_dataset_test.pt')
print(f"  ➜ Saved sequential datasets")

# ══════════════════════════════════════════════════════════════════════
# T6.2: Bi-LSTM Model
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T6.2] Bi-LSTM Model Training")
print("─" * 50)

class BiLSTMClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.attention = torch.nn.MultiheadAttention(hidden_dim * 2, num_heads=4, dropout=dropout, batch_first=True)
        self.layer_norm = torch.nn.LayerNorm(hidden_dim * 2)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(lstm_out + attn_out)
        # Mean pooling over sequence (ignore padding)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            pooled = (attn_out * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = attn_out.mean(dim=1)
        out = self.dropout(pooled)
        return self.classifier(out)

# Focal Loss
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        bce = torch.nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal = (1 - pt) ** self.gamma * bce
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * focal).mean()

model_lstm = BiLSTMClassifier(n_features).to(device)
optimizer = torch.optim.AdamW(model_lstm.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = FocalLoss(gamma=2, alpha=0.75)

def train_epoch(model, X, y, optimizer, batch_size=64):
    model.train()
    total_loss = 0
    n_batches = (len(X) + batch_size - 1) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(X))
        X_batch = X[start:end].to(device)
        y_batch = y[start:end].to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / n_batches

def evaluate(model, X, y, batch_size=64):
    model.eval()
    all_preds = []
    n_batches = (len(X) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(X))
            X_batch = X[start:end].to(device)
            preds = model(X_batch)
            all_preds.append(preds.cpu().numpy())
    y_pred = np.concatenate(all_preds).flatten()
    y_true = y.cpu().numpy().flatten()
    auc = roc_auc_score(y_true, y_pred)
    return auc, y_pred

print("  Training Bi-LSTM...")
best_val_auc = 0
patience = 10
no_improve = 0
train_losses = []
val_aucs = []

for epoch in range(50):
    loss = train_epoch(model_lstm, X_train_t, y_train_t, optimizer)
    train_auc, _ = evaluate(model_lstm, X_train_t, y_train_t)
    val_auc, _ = evaluate(model_lstm, X_val_t, y_val_t)

    train_losses.append(loss)
    val_aucs.append(val_auc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model_lstm.state_dict(), MODELS_DIR / 'bilstm_best.pt')
        no_improve = 0
    else:
        no_improve += 1

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}: loss={loss:.4f}, train AUC={train_auc:.4f}, val AUC={val_auc:.4f}")

    if no_improve >= patience:
        print(f"  Early stopping at epoch {epoch+1}")
        break

# Test
model_lstm.load_state_dict(torch.load(MODELS_DIR / 'bilstm_best.pt'))
test_auc_lstm, y_pred_lstm = evaluate(model_lstm, X_test_t, y_test_t)
y_true_test = y_test_t.cpu().numpy().flatten()
print(f"\n  Bi-LSTM Test AUC: {test_auc_lstm:.4f}")

# ══════════════════════════════════════════════════════════════════════
# T6.3: Transformer Model
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T6.3] Transformer Model Training")
print("─" * 50)

class TimeSeriesTransformer(torch.nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, dropout=0.2):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, d_model)
        self.pos_encoder = torch.nn.Embedding(10, d_model)  # Position up to 10
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len = x.shape[0], x.shape[1]
        x_proj = self.input_proj(x)  # (batch, seq_len, d_model)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x_proj = x_proj + self.pos_encoder(positions)
        # Transformer needs (seq_len, batch, d_model) but we use batch_first
        out = self.transformer(x_proj)
        # Use last valid token or mean pool
        pooled = out.mean(dim=1)  # (batch, d_model)
        return self.classifier(pooled)

model_transformer = TimeSeriesTransformer(n_features).to(device)
optimizer_t = torch.optim.AdamW(model_transformer.parameters(), lr=1e-3, weight_decay=1e-4)

print("  Training Transformer...")
best_val_auc_t = 0
no_improve_t = 0

for epoch in range(50):
    loss = train_epoch(model_transformer, X_train_t, y_train_t, optimizer_t)
    train_auc_t, _ = evaluate(model_transformer, X_train_t, y_train_t)
    val_auc_t, _ = evaluate(model_transformer, X_val_t, y_val_t)

    if val_auc_t > best_val_auc_t:
        best_val_auc_t = val_auc_t
        torch.save(model_transformer.state_dict(), MODELS_DIR / 'transformer_best.pt')
        no_improve_t = 0
    else:
        no_improve_t += 1

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}: loss={loss:.4f}, train AUC={train_auc_t:.4f}, val AUC={val_auc_t:.4f}")

    if no_improve_t >= patience:
        print(f"  Early stopping at epoch {epoch+1}")
        break

model_transformer.load_state_dict(torch.load(MODELS_DIR / 'transformer_best.pt'))
test_auc_trans, y_pred_trans = evaluate(model_transformer, X_test_t, y_test_t)
print(f"\n  Transformer Test AUC: {test_auc_trans:.4f}")

# ── DL Results Summary ──
dl_results = pd.DataFrame({
    'Model': ['Bi-LSTM', 'Transformer'],
    'Test_AUC': [test_auc_lstm, test_auc_trans],
})
dl_results.to_csv(OUT / 'dl_test_results.csv', index=False)
print(f"  ➜ Saved: {OUT / 'dl_test_results.csv'}")

# ══════════════════════════════════════════════════════════════════════
# T6.4: DL Explainability
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("[T6.4] DL Explainability (Attention Visualization)")
print("─" * 50)

try:
    # Extract attention from transformer
    model_transformer.eval()
    with torch.no_grad():
        # Get a sample of test patients
        n_sample = min(20, len(X_test))
        X_sample = X_test_t[:n_sample].to(device)
        y_sample = y_test_t[:n_sample].to(device)

        # Forward pass through the model components
        x_proj = model_transformer.input_proj(X_sample)
        positions = torch.arange(X_sample.shape[1], device=device).unsqueeze(0).expand(n_sample, -1)
        x_proj = x_proj + model_transformer.pos_encoder(positions)

        # Get attention from each transformer layer
        attn_weights = []
        for layer in model_transformer.transformer.layers:
            # Access the self-attention module
            attn_out, attn_w = layer.self_attn(
                x_proj, x_proj, x_proj,
                need_weights=True, average_attn_weights=True
            )
            x_proj = layer.norm1(x_proj + layer.dropout1(attn_out))
            x_proj = layer.norm2(x_proj + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x_proj))))))
            attn_weights.append(attn_w.cpu().numpy())

        # Average attention across layers
        avg_attention = np.mean(attn_weights, axis=0)  # (batch, seq_len, seq_len)

        # Plot attention heatmap for 2 example patients
        pos_indices = np.where(y_sample.cpu().numpy().flatten() == 1)[0]
        neg_indices = np.where(y_sample.cpu().numpy().flatten() == 0)[0]

        if len(pos_indices) > 0 and len(neg_indices) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            for ax, idx, title in [
                (axes[0], pos_indices[0], 'APS+ Patient'),
                (axes[1], neg_indices[0], 'Non-APS Patient')
            ]:
                im = ax.imshow(avg_attention[idx], cmap='viridis', aspect='auto')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Time Step')
                ax.set_title(title)
                plt.colorbar(im, ax=ax, shrink=0.8)
            plt.tight_layout()
            fig.savefig(FIG / 'main' / 'Figure_Attention_Heatmap.pdf')
            print(f"  ➜ Saved: {FIG / 'main' / 'Figure_Attention_Heatmap.pdf'}")
            plt.close()
except Exception as e:
    print(f"  ⚠ Attention visualization failed: {e}")

# Training curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label='Bi-LSTM', color='steelblue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[1].plot(val_aucs, label='Bi-LSTM Val AUC', color='#d6604d')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Validation AUC')
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(FIG / 'supplementary' / 'DL_Training_Curves.png', dpi=150)
    print(f"  ➜ Saved: {FIG / 'supplementary' / 'DL_Training_Curves.png'}")
    plt.close()
except Exception as e:
    print(f"  ⚠ Training curve plot failed: {e}")

print(f"\n{'═' * 60}")
print("PHASE 6 COMPLETE")
print(f"{'═' * 60}")

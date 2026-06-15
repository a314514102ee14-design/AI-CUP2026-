import argparse
import random
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

SEED = 42
PAD_TOKEN = 0
IGNORE_INDEX = -1

CAT_FEATURES = [
    "sex", "handId", "strengthId", "spinId",
    "pointId", "actionId", "positionId", "strikeId",
    "scoreSelf", "scoreOther"
]

MAX_STRIKE_DEFAULT = 40


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class RallyDataset(Dataset):
    def __init__(self, X_cat, X_num, yA_seq, yP_seq, yA_last, yP_last, yR, L):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.yA_seq = torch.tensor(yA_seq, dtype=torch.long)
        self.yP_seq = torch.tensor(yP_seq, dtype=torch.long)
        self.yA_last = torch.tensor(yA_last, dtype=torch.long)
        self.yP_last = torch.tensor(yP_last, dtype=torch.long)
        self.yR = torch.tensor(yR, dtype=torch.float32)
        self.L = torch.tensor(L, dtype=torch.long)

    def __len__(self):
        return self.X_cat.shape[0]

    def __getitem__(self, i):
        return (
            self.X_cat[i], self.X_num[i],
            self.yA_seq[i], self.yP_seq[i],
            self.yA_last[i], self.yP_last[i],
            self.yR[i], self.L[i]
        )


class CausalLastStepLSTM(nn.Module):
    def __init__(self, num_tokens_per_feature, n_act, n_pt,
                 emb_dim=24, hidden=256, num_layers=2, dropout=0.30,
                 num_dim=1, last_k=3):
        super().__init__()
        self.last_k = last_k

        self.embs = nn.ModuleList([
            nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN)
            for n in num_tokens_per_feature
        ])

        in_dim = len(num_tokens_per_feature) * emb_dim + num_dim

        self.lstm = nn.LSTM(
            in_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

        # last-step classification heads
        self.act_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_act)
        )
        self.pt_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_pt)
        )

        # rally-level head: mean + last + last-k pooling
        self.rly_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def last_k_pool(self, o, lengths):
        outs = []
        for i in range(o.size(0)):
            L = int(lengths[i].item())
            s = max(0, L - self.last_k)
            outs.append(o[i, s:L].mean(dim=0))
        return torch.stack(outs, dim=0)

    def forward(self, X_cat, X_num, lengths):
        es = [emb(X_cat[:, :, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(es + [X_num], dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        o, _ = self.lstm(packed)
        o, _ = nn.utils.rnn.pad_packed_sequence(
            o, batch_first=True, total_length=X_cat.size(1)
        )

        o = self.norm(o)
        o = self.drop(o)

        mask = (X_cat[:, :, 0] != PAD_TOKEN).float()
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_hidden = (o * mask.unsqueeze(-1)).sum(dim=1) / denom

        last_idx = (lengths - 1).clamp(min=0)
        last_hidden = o[torch.arange(o.size(0), device=o.device), last_idx]
        lastk_hidden = self.last_k_pool(o, lengths)

        rally_feat = torch.cat([mean_hidden, last_hidden, lastk_hidden], dim=-1)

        # Important: action/point logits are only for the final valid timestep
        act_last_logits = self.act_head(last_hidden)
        pt_last_logits = self.pt_head(last_hidden)
        rally_logits = self.rly_head(rally_feat).squeeze(1)

        return act_last_logits, pt_last_logits, rally_logits


def pad2d_cat(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
    out[:len(a)] = a
    return out


def pad2d_num(a, m):
    out = np.zeros((m, a.shape[1]), dtype=np.float32)
    out[:len(a)] = a
    return out


def pad1d(a, m, ignore_index=IGNORE_INDEX):
    out = np.full((m,), ignore_index, dtype=np.int64)
    out[:len(a)] = a
    return out


def map_with_ignore(arr, mapper):
    out = np.full_like(arr, IGNORE_INDEX, dtype=np.int64)
    mask = arr != IGNORE_INDEX
    out[mask] = np.array([mapper[int(v)] for v in arr[mask]], dtype=np.int64)
    return out


def make_class_weight(y, n_class, power=0.5):
    counts = np.bincount(y[y != IGNORE_INDEX].ravel(), minlength=n_class).astype(np.float32) + 1.0
    w = 1.0 / (counts ** power)
    w = w * (n_class / w.sum())
    return torch.tensor(w, dtype=torch.float32)


def main(args):
    seed_everything(args.seed)

    train = pd.read_csv(args.train).sort_values(["rally_uid", "strikeNumber"])
    test = pd.read_csv(args.test).sort_values(["rally_uid", "strikeNumber"])

    
    try:
        sample = pd.read_csv(args.sample)
    except Exception:
        sample = pd.DataFrame()

    train["strikeNumber"] = train["strikeNumber"].clip(0, args.max_strike)
    test["strikeNumber"] = test["strikeNumber"].clip(0, args.max_strike)

    cats = {c: pd.Categorical(train[c]).categories for c in CAT_FEATURES}

    def encode_cat(df):
        outs = []
        for col in CAT_FEATURES:
            codes = pd.Categorical(df[col], categories=cats[col]).codes + 1
            outs.append(np.asarray(codes, dtype=np.int64))
        return np.stack(outs, axis=1)

    def encode_num(df):
        strike_norm = (df["strikeNumber"].values.astype(np.float32) / float(args.max_strike)).reshape(-1, 1)
        return strike_norm

    Xc_list, Xn_list, yA_seq_list, yP_seq_list = [], [], [], []
    yA_last_list, yP_last_list, yR_list, L_list = [], [], [], []

    for _, g in train.groupby("rally_uid"):
        if len(g) < 2:
            continue

        Xc = encode_cat(g)[:-1]
        Xn = encode_num(g)[:-1]
        yA = g["actionId"].values[1:].astype(np.int64)
        yP = g["pointId"].values[1:].astype(np.int64)
        yR = int(g["serverGetPoint"].iloc[0])

        Xc_list.append(Xc)
        Xn_list.append(Xn)
        yA_seq_list.append(yA)
        yP_seq_list.append(yP)
        yA_last_list.append(yA[-1])
        yP_last_list.append(yP[-1])
        yR_list.append(yR)
        L_list.append(len(Xc))

    MAXLEN = max(L_list)

    Xc_all = np.stack([pad2d_cat(s, MAXLEN) for s in Xc_list])
    Xn_all = np.stack([pad2d_num(s, MAXLEN) for s in Xn_list])
    yA_seq_all = np.stack([pad1d(s, MAXLEN) for s in yA_seq_list])
    yP_seq_all = np.stack([pad1d(s, MAXLEN) for s in yP_seq_list])
    yA_last_all = np.array(yA_last_list, dtype=np.int64)
    yP_last_all = np.array(yP_last_list, dtype=np.int64)
    yR_all = np.array(yR_list, dtype=np.float32)
    L_all = np.array(L_list, dtype=np.int64)

    act_classes = np.sort(train["actionId"].unique())
    pt_classes = np.sort(train["pointId"].unique())
    n_act = len(act_classes)
    n_pt = len(pt_classes)

    act_id2idx = {v: i for i, v in enumerate(act_classes)}
    pt_id2idx = {v: i for i, v in enumerate(pt_classes)}

    yA_seq_all = map_with_ignore(yA_seq_all, act_id2idx)
    yP_seq_all = map_with_ignore(yP_seq_all, pt_id2idx)
    yA_last_all = np.array([act_id2idx[int(v)] for v in yA_last_all], dtype=np.int64)
    yP_last_all = np.array([pt_id2idx[int(v)] for v in yP_last_all], dtype=np.int64)

    idx = np.arange(len(Xc_all))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=(yR_all > 0.5)
    )

    Xc_tr, Xc_va = Xc_all[tr_idx], Xc_all[va_idx]
    Xn_tr, Xn_va = Xn_all[tr_idx], Xn_all[va_idx]
    yAseq_tr, yAseq_va = yA_seq_all[tr_idx], yA_seq_all[va_idx]
    yPseq_tr, yPseq_va = yP_seq_all[tr_idx], yP_seq_all[va_idx]
    yAlast_tr, yAlast_va = yA_last_all[tr_idx], yA_last_all[va_idx]
    yPlast_tr, yPlast_va = yP_last_all[tr_idx], yP_last_all[va_idx]
    yR_tr, yR_va = yR_all[tr_idx], yR_all[va_idx]
    L_tr, L_va = L_all[tr_idx], L_all[va_idx]

    # Weight 用 full sequence 統計比較穩；loss 只算 last-step
    act_w = make_class_weight(yAseq_tr, n_act, power=args.weight_power)
    pt_w = make_class_weight(yPseq_tr, n_pt, power=args.weight_power)

    pos = float((yR_tr == 1).sum())
    neg = float((yR_tr == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)

    train_ds = RallyDataset(Xc_tr, Xn_tr, yAseq_tr, yPseq_tr, yAlast_tr, yPlast_tr, yR_tr, L_tr)
    val_ds = RallyDataset(Xc_va, Xn_va, yAseq_va, yPseq_va, yAlast_va, yPlast_va, yR_va, L_va)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=max(args.batch * 2, 128), shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"Train rallies={len(train_ds)}, Val rallies={len(val_ds)}, MAXLEN={MAXLEN}")

    model = CausalLastStepLSTM(
        num_tokens_per_feature=[len(cats[c]) + 1 for c in CAT_FEATURES],
        n_act=n_act,
        n_pt=n_pt,
        emb_dim=args.emb,
        hidden=args.hidden,
        num_layers=args.layers,
        dropout=args.drop,
        num_dim=1,
        last_k=args.last_k
    ).to(device)

    ce_action = nn.CrossEntropyLoss(weight=act_w.to(device), label_smoothing=args.label_smoothing)
    ce_point = nn.CrossEntropyLoss(weight=pt_w.to(device), label_smoothing=args.label_smoothing)
    bce_rally = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    best_score = -1.0
    best_state = None
    bad_epochs = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0

        for Xcb, Xnb, yAseqb, yPseqb, yAlb, yPlb, yRb, Lb in train_loader:
            Xcb = Xcb.to(device)
            Xnb = Xnb.to(device)
            yAlb = yAlb.to(device)
            yPlb = yPlb.to(device)
            yRb = yRb.to(device)
            Lb = Lb.to(device)

            opt.zero_grad()
            la, lp, lr = model(Xcb, Xnb, Lb)

            loss_a = ce_action(la, yAlb)
            loss_p = ce_point(lp, yPlb)
            loss_r = bce_rally(lr, yRb)
            loss = args.loss_a * loss_a + args.loss_p * loss_p + args.loss_r * loss_r

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            run_loss += loss.item() * Xcb.size(0)

        model.eval()
        val_loss = 0.0
        allA, allAp, allP, allPp, allR, allRp = [], [], [], [], [], []

        with torch.no_grad():
            for Xcb, Xnb, yAseqb, yPseqb, yAlb, yPlb, yRb, Lb in val_loader:
                Xcb = Xcb.to(device)
                Xnb = Xnb.to(device)
                yAlb = yAlb.to(device)
                yPlb = yPlb.to(device)
                yRb = yRb.to(device)
                Lb = Lb.to(device)

                la, lp, lr = model(Xcb, Xnb, Lb)

                loss_a = ce_action(la, yAlb)
                loss_p = ce_point(lp, yPlb)
                loss_r = bce_rally(lr, yRb)
                loss = args.loss_a * loss_a + args.loss_p * loss_p + args.loss_r * loss_r
                val_loss += loss.item() * Xcb.size(0)

                allA += yAlb.detach().cpu().tolist()
                allAp += la.argmax(-1).detach().cpu().tolist()
                allP += yPlb.detach().cpu().tolist()
                allPp += lp.argmax(-1).detach().cpu().tolist()
                allR += yRb.detach().cpu().tolist()
                allRp += torch.sigmoid(lr).detach().cpu().tolist()

        tr_loss = run_loss / len(train_loader.dataset)
        va_loss = val_loss / len(val_loader.dataset)

        try:
            f1A = f1_score(allA, allAp, average="macro") if len(allA) else 0.0
            f1P = f1_score(allP, allPp, average="macro") if len(allP) else 0.0
            auc = roc_auc_score(allR, allRp) if len(set(allR)) > 1 else 0.5
        except Exception:
            f1A, f1P, auc = 0.0, 0.0, 0.5

        final = args.loss_a * f1A + args.loss_p * f1P + args.loss_r * auc
        scheduler.step(final)
        lr_now = opt.param_groups[0]["lr"]

        print(
            f"[Epoch {ep:02d}/{args.epochs}] "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"F1_action_last={f1A:.4f} F1_point_last={f1P:.4f} AUC={auc:.4f} "
            f"Final~{final:.4f} lr={lr_now:.2e}"
        )

        if final > best_score:
            best_score = final
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"Early stopping. Best Final~{best_score:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    print(f"Using best validation model: Final~{best_score:.4f}")

    pred_rows = []
    test_rids = []

    with torch.no_grad():
        for rid, g in test.groupby("rally_uid"):
            test_rids.append(int(rid))
            Xcg = encode_cat(g)
            Xng = encode_num(g)
            T = min(len(Xcg), MAXLEN)

            Xcp = pad2d_cat(Xcg[:T], MAXLEN)
            Xnp = pad2d_num(Xng[:T], MAXLEN)

            Xc_t = torch.tensor(Xcp[None, ...], dtype=torch.long, device=device)
            Xn_t = torch.tensor(Xnp[None, ...], dtype=torch.float32, device=device)
            L_t = torch.tensor([max(1, T)], dtype=torch.long, device=device)

            la, lp, lr = model(Xc_t, Xn_t, L_t)
            a_idx = int(torch.argmax(la[0]).item())
            p_idx = int(torch.argmax(lp[0]).item())
            s_prob = float(torch.sigmoid(lr).item())

            pred_rows.append({
                "rally_uid": int(rid),
                "serverGetPoint": s_prob,
                "pointId": int(pt_classes[p_idx]),
                "actionId": int(act_classes[a_idx])
            })

    pred_df = pd.DataFrame(pred_rows).sort_values("rally_uid")

    # Robust submission output:
    # 1) sample 有列數且含 rally_uid -> 依 sample 對齊
    # 2) sample 是空的 -> 直接用 test rally_uid 輸出
    if (not sample.empty) and ("rally_uid" in sample.columns):
        out = sample.copy()
        pred_map = pred_df.set_index("rally_uid")
        for col in ["serverGetPoint", "pointId", "actionId"]:
            out[col] = out["rally_uid"].map(pred_map[col])
        out["serverGetPoint"] = out["serverGetPoint"].fillna(0.5)
        out["pointId"] = out["pointId"].fillna(int(pt_classes[0])).astype(int)
        out["actionId"] = out["actionId"].fillna(int(act_classes[0])).astype(int)
        out = out[sample.columns]
    else:
        out = pred_df[["rally_uid", "serverGetPoint", "pointId", "actionId"]].copy()

    out.to_csv(args.out, index=False)
    print(f"Saved submission to: {args.out}")
    print(out.head())
    print("Rows:", len(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test_new.csv")
    ap.add_argument("--sample", default="sample_submission.csv")
    ap.add_argument("--out", default="submission_laststep.csv")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--emb", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--drop", type=float, default=0.30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--last_k", type=int, default=3)

    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max_strike", type=int, default=MAX_STRIKE_DEFAULT)

    ap.add_argument("--loss_a", type=float, default=0.40)
    ap.add_argument("--loss_p", type=float, default=0.40)
    ap.add_argument("--loss_r", type=float, default=0.20)

    ap.add_argument("--weight_power", type=float, default=0.5)
    ap.add_argument("--label_smoothing", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    args = ap.parse_args()
    main(args)

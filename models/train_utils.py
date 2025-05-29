import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

def rand_bbox(size, lam):
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 중심점 무작위 선택
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class EarlyStopping:
    def __init__(self, patience=3, verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.verbose = verbose
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"📉 EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def multiclass_log_loss(answer_df, submission_df):
    # 클래스 리스트를 문자열로 정리
    class_list = sorted([str(label) for label in answer_df['label'].unique()])

    # ID 기준 정렬
    submission_df = submission_df.sort_values(by='ID').reset_index(drop=True)
    answer_df = answer_df.sort_values(by='ID').reset_index(drop=True)

    # ID 일치 검증
    if not all(answer_df['ID'] == submission_df['ID']):
        raise ValueError("ID가 정렬되지 않았거나 불일치합니다.")

    # label도 문자열로 변환
    answer_df['label'] = answer_df['label'].astype(str)

    # 제출 컬럼 검증
    missing_cols = [col for col in class_list if col not in submission_df.columns]
    if missing_cols:
        raise ValueError(f"클래스 컬럼 누락: {missing_cols}")

    if submission_df[class_list].isnull().any().any():
        raise ValueError("NaN 포함됨")

    for col in class_list:
        if not ((submission_df[col] >= 0) & (submission_df[col] <= 1)).all():
            raise ValueError(f"{col}의 확률값이 0~1 범위 초과")

    # 정답 인덱스 변환
    true_labels = answer_df['label'].tolist()
    true_idx = [class_list.index(lbl) for lbl in true_labels]

    # 확률 정규화 + clip
    probs = submission_df[class_list].values
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_pred = np.clip(probs, 1e-15, 1 - 1e-15)

    return log_loss(true_idx, y_pred, labels=list(range(len(class_list))))

def train_one_epoch(model, dataloader, criterion, optimizer, device, mixup_fn=None):
    model.train()
    total_loss = 0
    for x, y, *_ in tqdm(dataloader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)

        if mixup_fn is not None:
            x, y = mixup_fn(x, y) # y는 soft-label(one-hot)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    probs, preds, labels, ids, img_paths = [], [], [], [], []

    with torch.no_grad():
        for i, (x, y, paths) in enumerate(tqdm(dataloader, desc="Eval", leave=False)):
            x = x.to(device)
            out = model(x)
            prob = torch.softmax(out, dim=1).cpu().numpy()

            preds += list(np.argmax(prob, axis=1))
            probs.append(prob)
            labels += y.numpy().tolist()
            ids += list(range(i * dataloader.batch_size, i * dataloader.batch_size + len(x)))
            img_paths += paths

    return preds, np.vstack(probs), labels, ids, img_paths



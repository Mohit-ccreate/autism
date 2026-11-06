"""
Heinsfeld-Style Deep Autoencoder Classifier for Neuroimaging Connectomes.
src/models/autoencoder.py

Re-implementation of the deep autoencoder architecture from:
Heinsfeld et al. (2018), "Identification of autism spectrum disorder using deep neural networks applied to fMRI data", NeuroImage: Clinical.

Architecture:
1. Deep Stacked Autoencoder for unsupervised representation learning on resting-state connectivity matrices.
2. Supervised fine-tuning of the latent representations for binary classification (ASD vs Control).
3. Strictly fitted inside training folds only (zero cross-fold or test-site leakage).

Compatible with the scikit-learn Classifier API.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin


class AutoencoderNet(nn.Module):
    """Deep autoencoder with classification head."""
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_recon(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def forward_clf(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.classifier(z).squeeze(-1)


class HeinsfeldAutoencoderClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible classifier implementing the Heinsfeld et al. (2018) autoencoder pipeline.
    """
    def __init__(
        self,
        latent_dim: int = 32,
        pretrain_epochs: int = 25,
        finetune_epochs: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        seed: int = 42,
    ):
        self.latent_dim = latent_dim
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seed = seed
        self.model: nn.Module | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        self.classes_ = np.unique(y_arr)
        input_dim = X_arr.shape[1]

        self.model = AutoencoderNet(input_dim=input_dim, latent_dim=self.latent_dim)
        device = torch.device("cpu")
        self.model.to(device)

        # Stage 1: Unsupervised Reconstruction Pre-training
        dataset_recon = TensorDataset(torch.from_numpy(X_arr))
        loader_recon = DataLoader(dataset_recon, batch_size=min(self.batch_size, len(X_arr)), shuffle=True, drop_last=(len(X_arr) > self.batch_size))
        opt_recon = optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        recon_loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.pretrain_epochs):
            for (bx,) in loader_recon:
                bx = bx.to(device)
                opt_recon.zero_grad()
                x_recon = self.model.forward_recon(bx)
                loss = recon_loss_fn(x_recon, bx)
                loss.backward()
                opt_recon.step()

        # Stage 2: Supervised Fine-tuning for ASD Classification
        dataset_clf = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr))
        loader_clf = DataLoader(dataset_clf, batch_size=min(self.batch_size, len(X_arr)), shuffle=True, drop_last=(len(X_arr) > self.batch_size))
        opt_clf = optim.Adam(
            self.model.parameters(),
            lr=self.lr * 0.5,
            weight_decay=self.weight_decay,
        )
        clf_loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(self.finetune_epochs):
            for bx, by in loader_clf:
                bx, by = bx.to(device), by.to(device)
                opt_clf.zero_grad()
                logits = self.model.forward_clf(bx)
                loss = clf_loss_fn(logits, by)
                loss.backward()
                opt_clf.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        self.model.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            bx = torch.from_numpy(X_arr)
            logits = self.model.forward_clf(bx)
            p1 = torch.sigmoid(logits).numpy()
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return (prob >= 0.5).astype(int)

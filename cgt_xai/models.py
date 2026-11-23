# train_model.py

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Torch RNN wrapper (sklearn-like API)
def _set_all_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


class TorchRNNClassifier:
    """Treat tabular X (n, d) as a sequence (n, d, 1) and train a gru."""

    def __init__(
        self,
        seed: int = 42,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        max_epochs: int = 30,
        patience: int = 5,
        device: str | None = None,
        verbose: bool = False,
    ):
        self.seed = seed
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.verbose = verbose

        self.classes_ = None
        self._model = None
        self._mu = None
        self._sigma = None
        self._device = None

    def _prep_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X = (X - self._mu) / self._sigma
        return X[..., None]

    def fit(self, X: np.ndarray, y: np.ndarray):
        _set_all_seeds(self.seed)

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        cls_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([cls_to_idx[yy] for yy in y], dtype=np.int64)

        self._mu = X.mean(axis=0, keepdims=True)
        self._sigma = X.std(axis=0, keepdims=True) + 1e-6

        Xseq = self._prep_X(X)

        if self.device is not None:
            self._device = self.device
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        n_classes = len(self.classes_)

        class GRUNet(nn.Module):
            def __init__(self, hidden_size, num_layers, dropout, n_classes):
                super().__init__()
                self.rnn = nn.GRU(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.fc = nn.Linear(hidden_size, n_classes)

            def forward(self, x):
                out, _ = self.rnn(x)
                last = out[:, -1, :]
                return self.fc(last)

        self._model = GRUNet(self.hidden_size, self.num_layers, self.dropout, n_classes).to(self._device)

        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        ds = TensorDataset(torch.from_numpy(Xseq), torch.from_numpy(y_idx))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        best_loss = float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(self.max_epochs):
            self._model.train()
            total = 0.0
            n_seen = 0

            for xb, yb in dl:
                xb = xb.to(self._device)
                yb = yb.to(self._device)

                optim.zero_grad(set_to_none=True)
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optim.step()

                total += float(loss.item()) * xb.size(0)
                n_seen += xb.size(0)

            epoch_loss = total / max(1, n_seen)

            if self.verbose:
                print(f"[RNN] epoch {epoch+1}/{self.max_epochs} loss={epoch_loss:.4f}")

            # Early stopping on train loss (simple + stable under repeated retrains)
            if epoch_loss + 1e-6 < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        X = np.asarray(X, dtype=np.float32)
        Xseq = self._prep_X(X)

        self._model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(Xseq).to(self._device)
            logits = self._model(xb)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]


class TorchLSTMClassifier:
    """Treat tabular X (n, d) as a sequence (n, d, 1) and train an lstm.
    
        Exposes:
          - fit(X, y)
          - predict(X)
          - predict_proba(X)
          - classes_
    """

    def __init__(
        self,
        seed: int = 42,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        max_epochs: int = 30,
        patience: int = 5,
        device: str | None = None,
        verbose: bool = False,
    ):
        self.seed = seed
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.verbose = verbose

        self.classes_ = None
        self._model = None
        self._mu = None
        self._sigma = None
        self._device = None

    def _prep_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X = (X - self._mu) / self._sigma
        return X[..., None]

    def fit(self, X: np.ndarray, y: np.ndarray):
        _set_all_seeds(self.seed)

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        cls_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([cls_to_idx[yy] for yy in y], dtype=np.int64)

        self._mu = X.mean(axis=0, keepdims=True)
        self._sigma = X.std(axis=0, keepdims=True) + 1e-6

        Xseq = self._prep_X(X)

        if self.device is not None:
            self._device = self.device
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        n_classes = len(self.classes_)

        class LSTMNet(nn.Module):
            def __init__(self, hidden_size, num_layers, dropout, n_classes):
                super().__init__()
                self.rnn = nn.LSTM(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.fc = nn.Linear(hidden_size, n_classes)

            def forward(self, x):
                out, _ = self.rnn(x)
                last = out[:, -1, :]
                return self.fc(last)

        self._model = LSTMNet(self.hidden_size, self.num_layers, self.dropout, n_classes).to(self._device)

        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        ds = TensorDataset(torch.from_numpy(Xseq), torch.from_numpy(y_idx))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        best_loss = float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(self.max_epochs):
            self._model.train()
            total = 0.0
            n_seen = 0

            for xb, yb in dl:
                xb = xb.to(self._device)
                yb = yb.to(self._device)

                optim.zero_grad(set_to_none=True)
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optim.step()

                total += float(loss.item()) * xb.size(0)
                n_seen += xb.size(0)

            epoch_loss = total / max(1, n_seen)

            if self.verbose:
                print(f"[LSTM] epoch {epoch+1}/{self.max_epochs} loss={epoch_loss:.4f}")

            # Early stopping on train loss (simple + stable under repeated retrains)
            if epoch_loss + 1e-6 < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        X = np.asarray(X, dtype=np.float32)
        Xseq = self._prep_X(X)

        self._model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(Xseq).to(self._device)
            logits = self._model(xb)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]

# Model factory
def get_model(model_name: str = "logistic", random_state: int = 42):
    model_name = model_name.lower()

    if model_name == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=random_state, max_iter=1000))
        ])

    elif model_name == "random_forest":
        return RandomForestClassifier(random_state=random_state)

    elif model_name == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=random_state))
        ])

    elif model_name == "rnn":
        # Reasonable defaults for repeated retraining inside v(S)
        return TorchRNNClassifier(
            seed=random_state,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            lr=1e-3,
            weight_decay=0.0,
            batch_size=64,
            max_epochs=20,
            patience=3,
            verbose=False,
        )

    elif model_name == "lstm":
        return TorchLSTMClassifier(
            seed=random_state,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            lr=1e-3,
            weight_decay=0.0,
            batch_size=64,
            max_epochs=20,
            patience=3,
            verbose=False,
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


# Train + save helper
def train_and_save_model(model_name, X_train, y_train, save_path="trained_model.joblib"):
    model = get_model(model_name)
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")
    return model

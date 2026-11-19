"""Train and refit the PyTorch MLP."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import CardioMLP


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_probabilities(model, features):
    model.eval()
    tensor = torch.as_tensor(features, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(model(tensor)).cpu().numpy().reshape(-1)


def _loader(features, target, batch_size, seed):
    dataset = TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(target, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    drop_last = len(dataset) % batch_size == 1
    return dataset, DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        generator=generator,
    )


def _loss_function(target, positive_weight_multiplier=1.3):
    positives = max(int(np.sum(target == 1)), 1)
    negatives = max(int(np.sum(target == 0)), 1)
    weight = torch.tensor([negatives / positives * positive_weight_multiplier], dtype=torch.float32)
    return torch.nn.BCEWithLogitsLoss(pos_weight=weight)


def _make_optimizer(name, parameters, learning_rate, weight_decay):
    key = name.lower()
    if key == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if key == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if key == "sgd":
        return torch.optim.SGD(
            parameters, lr=learning_rate, momentum=0.9,
            nesterov=True, weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def train_with_early_stopping(
    x_train,
    y_train,
    x_valid,
    y_valid,
    seed,
    batch_size,
    max_epochs,
    patience,
):
    seed_everything(seed)
    model = CardioMLP(x_train.shape[1])
    criterion = _loss_function(y_train)
    optimizer = _make_optimizer("adamw", model.parameters(), 1e-3, 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5,
    )
    train_data, train_loader = _loader(x_train, y_train, batch_size, seed)
    valid_data = TensorDataset(
        torch.as_tensor(x_valid, dtype=torch.float32),
        torch.as_tensor(y_valid, dtype=torch.float32),
    )
    valid_loader = DataLoader(valid_data, batch_size=2048, shuffle=False)
    best_loss = float("inf")
    best_epoch = 1
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                loss = criterion(model(batch_x), batch_y)
                validation_loss += loss.item() * len(batch_x)
        validation_loss /= len(valid_data)
        train_loss /= len(train_data)
        scheduler.step(validation_loss)
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "validation_loss": float(validation_loss),
        })
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss, history


def train_fixed_epochs(
    features, target, epochs, seed, batch_size, dropout=0.3,
    hidden_dims=(128, 64, 32), positive_weight_multiplier=1.3,
    optimizer_name="adamw", learning_rate=1e-3, weight_decay=1e-4,
):
    seed_everything(seed)
    model = CardioMLP(
        features.shape[1], dropout=dropout, hidden_dims=hidden_dims
    )
    criterion = _loss_function(target, positive_weight_multiplier)
    optimizer = _make_optimizer(
        optimizer_name, model.parameters(), learning_rate, weight_decay
    )
    _, loader = _loader(features, target, batch_size, seed)
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model

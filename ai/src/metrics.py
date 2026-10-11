"""Métricas sin dependencias de entrenamiento de alto nivel."""
import numpy as np
import torch


def metrics_from_confusion(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    true_positive = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted != 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision),
                   where=precision + recall != 0)
    return {"accuracy": float(true_positive.sum() / max(1, confusion.sum())),
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
            "per_class": [{"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                          for p, r, f, s in zip(precision, recall, f1, support)]}


def run_epoch(model, loader, class_weights, device, optimizer=None, augmentation=None, generator=None):
    from data import spec_augment
    training = optimizer is not None
    model.train(training)
    count = class_weights.numel()
    confusion = torch.zeros(count, count, dtype=torch.int64, device=device)
    loss_sum = weight_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="sum")
    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            if training and augmentation is not None:
                inputs = spec_augment(inputs, augmentation, generator)
            inputs, labels = inputs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            numerator = criterion(logits, labels)
            denominator = class_weights[labels].sum()
            loss = numerator / denominator
            if not torch.isfinite(loss):
                raise FloatingPointError("Pérdida no finita")
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += numerator.detach().item()
            weight_sum += denominator.item()
            predictions = logits.detach().argmax(dim=1)
            confusion += torch.bincount(labels * count + predictions, minlength=count * count).reshape(count, count)
    if weight_sum == 0:
        raise ValueError("Partición vacía")
    matrix = confusion.cpu().numpy()
    metrics = metrics_from_confusion(matrix)
    metrics["loss"] = loss_sum / weight_sum
    return metrics, matrix

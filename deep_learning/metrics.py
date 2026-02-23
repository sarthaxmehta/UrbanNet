import torch


def dice_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = target.float()

    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-6)


def iou_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return intersection / (union + 1e-6)

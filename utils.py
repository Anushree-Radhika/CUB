import re
import torch

def classification_accuracy(generated_texts, ground_truths):

    correct = 0
    total = len(ground_truths)

    for generated, gt in zip(generated_texts, ground_truths):

        # Accept either single or double quotes
        match = re.search(r"""["']([^"']+)["']""", generated)

        if match is None:
            continue

        predicted = match.group(1).strip().lower()
        gt = gt.strip().strip("'\"").strip().lower()
        
        # also replace underscores with spaces just in case the model generates them that way
        predicted = predicted.replace("_", " ")
        gt = gt.replace("_", " ")

        if predicted == gt or predicted in gt or gt in predicted:
            correct += 1

    return correct / total if total > 0 else 0.0

class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(self.count, 1)

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
            if scheduler is not None
            else None,
        },
        path,
    )


##############################################################
# Load Checkpoint
##############################################################

def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
):

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    if (
        scheduler is not None
        and checkpoint["scheduler"] is not None
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

    return checkpoint["epoch"]

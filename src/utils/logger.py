import csv, os
from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    def __init__(self, outdir: str):
        os.makedirs(outdir, exist_ok=True)
        self.csv_path = os.path.join(outdir, "training_log.csv")
        self.tb = SummaryWriter(log_dir=os.path.join(outdir, "tb"))
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch","train_loss","val_pr_auc"])

    def log_epoch(self, epoch: int, train_loss: float, val_pr_auc: float):
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{train_loss:.6f}", f"{val_pr_auc:.6f}"])
        self.tb.add_scalar("loss/train", train_loss, epoch)
        self.tb.add_scalar("val/pr_auc_illicit", val_pr_auc, epoch)

    def close(self):
        self.tb.flush()
        self.tb.close()

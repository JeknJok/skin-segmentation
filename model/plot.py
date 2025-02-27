import matplotlib.pyplot as plt
import tensorflow as tf

class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.losses = []
        self.val_losses = []
        self.mean_iou = []
        self.val_iou = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.mean_iou.append(logs.get("mean_iou"))
        self.val_iou.append(logs.get("val_mean_iou"))

        plt.clf()

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.losses, label="Train Loss", color="red")
        plt.plot(self.val_losses, label="Val Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()

        # IoU Score
        plt.subplot(1, 2, 2)
        plt.plot(self.mean_iou, label="Train IoU", color="blue")
        plt.plot(self.val_iou, label="Val IoU", color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("IoU Score")
        plt.title("Training & Validation IoU Score")
        plt.legend()

        # Save plot
        plt.tight_layout()
        plt.savefig(self.save_path)

# utils/utils.py
import matplotlib.pyplot as plt
import torch


def plot_loss(train_losses, output_path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

# main.py
import torch
from config import Config

if __name__ == "__main__":
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cfg.trainer(device = device)
    print("Training finished. Model saved to outputs/")

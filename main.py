# main.py
import torch
from config import Config
from trainers.train import train

if __name__ == "__main__":
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, acc = train(cfg, device=device)
    print("Training finished. Model saved to outputs/")

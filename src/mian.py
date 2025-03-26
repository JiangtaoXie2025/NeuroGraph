# main.py
#
# Entry point for training, evaluating, or running inference with the model.

import argparse
import torch
from torch.utils.data import DataLoader
from src.data_loaders.brain_dataset import BrainDataset
from src.models.ngan import NeuroGraphAttentionNetwork
from src.utils.helpers import get_device, setup_logger
from src.utils.metrics import mpjpe, pck

def parse_args():
    parser = argparse.ArgumentParser(description="Main entry point for training or evaluation")
    parser.add_argument("--data-path", type=str, default="data/processed", help="Path to dataset")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--log-dir", type=str, default="outputs/logs", help="Directory for logs")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints", help="Directory for checkpoints")
    return parser.parse_args()

def train():
    args = parse_args()
    device = get_device()
    logger = setup_logger(args.log_dir)

    dataset = BrainDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = NeuroGraphAttentionNetwork(in_features=128, hidden_dim=64, num_classes=10)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            # Example batch retrieval, depends on how you structure your data
            # Suppose batch has 'input' and 'label'
            input_data = batch["input"].to(device)
            adjacency = batch.get("adjacency", None)
            if adjacency is not None:
                adjacency = adjacency.to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_data, adjacency)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}")

def evaluate():
    args = parse_args()
    # Load the trained model, create DataLoader, measure metrics, etc.
    pass

if __name__ == "__main__":
    train()

import torch
import torch.nn as nn
import torch.optim as optim
import time


def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-3, device='cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    print(f"\nTraining on device: {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 60)

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        running_train_loss = 0.0
        num_train_batches = 0

        for images, _ in train_loader:
            images = images.to(device)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = running_train_loss / num_train_batches
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                running_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = running_val_loss / num_val_batches
        val_losses.append(avg_val_loss)

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Time: {elapsed:.1f}s")

    print("-" * 60)
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"Final Val Loss:   {val_losses[-1]:.6f}")

    return train_losses, val_losses

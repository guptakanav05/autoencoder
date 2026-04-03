import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from data_loader import get_data_loaders
from models import get_model, MODEL_INFO
from train import train_model
from visualize import (
    plot_reconstructions,
    plot_loss_curves,
    plot_all_losses_comparison,
    plot_latent_space,
    print_summary_table,
)


def main():
    BATCH_SIZE = 128
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("STEP 1: Loading EMNIST Dataset")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=BATCH_SIZE,
        val_split=0.2
    )

    sample_images, sample_labels = next(iter(train_loader))
    print(f"\nImage shape: {sample_images.shape}")
    print(f"Pixel range: [{sample_images.min():.1f}, {sample_images.max():.1f}]")

    model_names = ['fc_v1', 'fc_v2', 'cnn_v1', 'cnn_v2']
    all_results = {}

    for model_name in model_names:
        info = MODEL_INFO[model_name]

        print("\n" + "=" * 60)
        print(f"STEP 2: Training {info['name']}")
        print(f"  Type: {info['type']}")
        print(f"  Bottleneck: {info['bottleneck']}")
        print(f"  Architecture: {info['description']}")
        print("=" * 60)

        model = get_model(model_name)

        train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            device=device
        )

        all_results[model_name] = {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'info': info,
        }

        checkpoint_path = os.path.join(RESULTS_DIR, f'{model_name}_model.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  Saved model checkpoint: {checkpoint_path}")

    print("\n" + "=" * 60)
    print("STEP 3: Generating Visualizations")
    print("=" * 60)

    for model_name, result in all_results.items():
        model = result['model']
        info = result['info']

        print(f"\n--- {info['name']} ---")

        plot_reconstructions(
            model=model,
            data_loader=test_loader,
            model_name=model_name,
            device=device,
            num_images=10,
            results_dir=RESULTS_DIR
        )

        plot_loss_curves(
            train_losses=result['train_losses'],
            val_losses=result['val_losses'],
            model_name=info['name'],
            results_dir=RESULTS_DIR
        )

        plot_latent_space(
            model=model,
            data_loader=test_loader,
            model_name=info['name'],
            device=device,
            num_samples=2000,
            results_dir=RESULTS_DIR
        )

    print("\n--- All Models Comparison ---")
    plot_all_losses_comparison(all_results, results_dir=RESULTS_DIR)

    print_summary_table(all_results)

    print(f"\nAll plots saved to '{RESULTS_DIR}/' directory.")
    print("Done!")


if __name__ == '__main__':
    main()

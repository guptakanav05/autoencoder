import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def ensure_results_dir(results_dir='results'):
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def plot_reconstructions(model, data_loader, model_name, device='cpu',
                         num_images=10, results_dir='results'):
    ensure_results_dir(results_dir)
    model.eval()

    images, labels = next(iter(data_loader))
    images = images[:num_images].to(device)

    with torch.no_grad():
        reconstructed = model(images)

    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    fig, axes = plt.subplots(2, num_images, figsize=(20, 4))
    fig.suptitle(f'{model_name} - Input vs Reconstructed', fontsize=14, fontweight='bold')

    for i in range(num_images):
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(results_dir, f'{model_name}_reconstructions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved reconstruction plot: {save_path}")


def plot_loss_curves(train_losses, val_losses, model_name, results_dir='results'):
    ensure_results_dir(results_dir)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', markersize=4)
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'{model_name} - Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(results_dir, f'{model_name}_loss_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved loss curves: {save_path}")


def plot_all_losses_comparison(all_results, results_dir='results'):
    ensure_results_dir(results_dir)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (model_name, result) in enumerate(all_results.items()):
        epochs = range(1, len(result['train_losses']) + 1)
        color = colors[idx % len(colors)]

        axes[0].plot(epochs, result['train_losses'], '-o', color=color,
                     label=model_name, markersize=3, linewidth=1.5)
        axes[1].plot(epochs, result['val_losses'], '-o', color=color,
                     label=model_name, markersize=3, linewidth=1.5)

    axes[0].set_title('Training Loss Comparison', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Validation Loss Comparison', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('All Models - Loss Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(results_dir, 'all_models_loss_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison plot: {save_path}")


def plot_latent_space(model, data_loader, model_name, device='cpu',
                      num_samples=2000, results_dir='results'):
    ensure_results_dir(results_dir)
    model.eval()

    all_latents = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            latent = model.encode(images)
            all_latents.append(latent.cpu().numpy())
            all_labels.append(labels.numpy())

            if sum(len(l) for l in all_latents) >= num_samples:
                break

    all_latents = np.concatenate(all_latents, axis=0)[:num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:num_samples]

    print(f"  Running t-SNE for {model_name} (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    latent_2d = tsne.fit_transform(all_latents)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(all_labels)
    try:
        cmap = plt.colormaps.get_cmap('tab20')
    except AttributeError:
        cmap = plt.cm.get_cmap('tab20', len(unique_labels))

    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                         c=all_labels, cmap='tab20',
                         s=5, alpha=0.6)

    plt.colorbar(scatter, label='Character Class')
    plt.title(f'{model_name} - Latent Space (t-SNE)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    save_path = os.path.join(results_dir, f'{model_name}_latent_space.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved latent space plot: {save_path}")


def print_summary_table(all_results):
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<25} {'Type':<18} {'Bottleneck':<12} {'Final Train':<14} {'Final Val':<14}")
    print("-" * 80)

    for model_name, result in all_results.items():
        info = result['info']
        final_train = result['train_losses'][-1] if result['train_losses'] else 'N/A (loaded)'
        final_val = result['val_losses'][-1] if result['val_losses'] else 'N/A (loaded)'

        if isinstance(final_train, float):
            print(f"{info['name']:<25} {info['type']:<18} {info['bottleneck']:<12} "
                  f"{final_train:<14.6f} {final_val:<14.6f}")
        else:
            print(f"{info['name']:<25} {info['type']:<18} {info['bottleneck']:<12} "
                  f"{final_train:<14} {final_val:<14}")

    print("=" * 80)

    models_with_losses = {k: v for k, v in all_results.items() if v['val_losses']}
    if models_with_losses:
        best_model = min(models_with_losses.items(), key=lambda x: x[1]['val_losses'][-1])
        print(f"\nBest model (lowest validation loss): {best_model[1]['info']['name']}")
        print(f"  Final validation loss: {best_model[1]['val_losses'][-1]:.6f}")
    else:
        print("\nModels loaded from checkpoints - no loss data available.")

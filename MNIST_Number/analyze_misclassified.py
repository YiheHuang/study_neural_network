import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def tta_predict(model, data, tta_times):
    logits_sum = model(data)
    for _ in range(tta_times - 1):
        angle, translations, scale, shear = transforms.RandomAffine.get_params(
            degrees=(-10, 10),
            translate=(0.08, 0.08),
            scale_ranges=None,
            shears=None,
            img_size=[28, 28],
        )
        augmented = TF.affine(
            data,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
        )
        logits_sum += model(augmented)
    return logits_sum / tta_times


def denormalize(x):
    return x * 0.3081 + 0.1307


def main():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "..", "mnist_cnn_best.pth")
    output_png = os.path.join(base_dir, "misclassified_samples.png")
    output_txt = os.path.join(base_dir, "misclassified_summary.txt")

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_dataset = datasets.MNIST(
        root=os.path.join(base_dir, "data"),
        train=False,
        download=True,
        transform=test_transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    misclassified = []
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    tta_times = 5
    sample_index = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = tta_predict(model, data, tta_times)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            confidence = probs.max(dim=1).values

            for i in range(data.size(0)):
                t = int(target[i].item())
                p = int(pred[i].item())
                confusion[t, p] += 1
                if p != t:
                    misclassified.append(
                        {
                            "idx": sample_index + i,
                            "true": t,
                            "pred": p,
                            "conf": float(confidence[i].item()),
                            "image": data[i].detach().cpu(),
                        }
                    )
            sample_index += data.size(0)

    misclassified.sort(key=lambda x: x["conf"], reverse=True)
    top_k = min(25, len(misclassified))
    rows, cols = 5, 5
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(rows * cols):
        axes[i].axis("off")
        if i < top_k:
            item = misclassified[i]
            img = denormalize(item["image"]).squeeze(0).clamp(0, 1)
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(
                f"idx={item['idx']}\nT:{item['true']} P:{item['pred']} ({item['conf']:.2f})",
                fontsize=8,
            )

    fig.suptitle("Top confident misclassified samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close(fig)

    pair_stats = []
    for t in range(10):
        for p in range(10):
            if t != p and confusion[t, p] > 0:
                pair_stats.append((int(confusion[t, p].item()), t, p))
    pair_stats.sort(reverse=True)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"Total test samples: {len(test_dataset)}\n")
        f.write(f"Total misclassified: {len(misclassified)}\n")
        f.write(f"Accuracy: {(1 - len(misclassified) / len(test_dataset)) * 100:.2f}%\n\n")
        f.write("Top-10 confusion pairs (true -> pred):\n")
        for count, t, p in pair_stats[:10]:
            f.write(f"  {t} -> {p}: {count}\n")
        f.write("\nTop-20 most confident errors:\n")
        for item in misclassified[:20]:
            f.write(
                f"  idx={item['idx']}, true={item['true']}, pred={item['pred']}, conf={item['conf']:.4f}\n"
            )

    print(f"Total misclassified: {len(misclassified)}")
    print(f"Saved image grid: {output_png}")
    print(f"Saved summary: {output_txt}")


if __name__ == "__main__":
    main()

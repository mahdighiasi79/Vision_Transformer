import argparse

# import pytest
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import optim

import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

from torchvision.transforms import RandomResizedCrop

from models import ViT, CrossViT  # rename the skeleton file for your implementation / comment before testing for ResNet

validation_set_accuracies = []


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to classify CIFAR10')
    parser.add_argument('--model', type=str, default='cvit', help='model to train (default: r18)')
    parser.add_argument('--batch-size', type=int, default=256, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 0.003)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay for optimizer (default: 0.01)')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='weight decay for optimizer (default: 0.01)')
    return parser.parse_args()


# a helper function to schedule the learning rate of the AdamW optimizer for ViT and CrossViT
def adjust_learning_rate(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        # Linear warmup
        lr = args.lr * (epoch + 1) / args.warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        # Use cosine schedule after warmup
        progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr


# a helper function to apply MixUp online regularizer
def mixup_data(x, y, alpha=0.8):
    """Returns mixed inputs, mixed labels, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# a helper function to apply CutMix online regularizer
def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix to a batch."""
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(x.device)

    # bounding box
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply CutMix patch
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # adjust lambda based on actual area
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def train(model, trainloader, optimizer, criterion, device, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)

        # Choose online augmentation
        use_mixup = np.random.rand() < 0.5  # 50% chance Mixup
        use_cutmix = np.random.rand() < 0.5  # 50% chance CutMix

        if use_mixup:
            data, y_a, y_b, lam = mixup_data(data, target, alpha=0.8)
        elif use_cutmix:
            data, y_a, y_b, lam = cutmix_data(data, target, alpha=1.0)
        else:
            y_a, y_b, lam = target, target, 1.0  # no mixing

        optimizer.zero_grad()
        output = model(data)
        loss = (lam * criterion(output, y_a)) + ((1 - lam) * criterion(output, target))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item()))
            if args.dry_run:
                break


def run_test(model, device, test_loader, criterion, set="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        set, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    validation_set_accuracies.append(100. * correct / len(test_loader.dataset))


def run(args, use_saved_models):
    # Download and load the training data
    transform = transforms.Compose([
        # ImageNet mean/std values should also fit okayish for CIFAR
        RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33))
    ])

    # TODO: adjust folder
    dataset = datasets.CIFAR10('..\\Exercise01 - ViT and Cross ViT\\train_images', download=True, train=True,
                               transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset,
                                                     [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    # Download and load the test data
    # TODO: adjust folder
    testset = datasets.CIFAR10('..\\Exercise01 - ViT and Cross ViT\\test_images', download=True, train=False,
                               transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=True)

    # Build a feed-forward network
    print(f"Using {args.model}")
    if args.model == "r18":
        model = models.resnet18(pretrained=False)
    elif args.model == "vit":
        model = ViT(image_size=32, patch_size=8, num_classes=10, dim=64,
                    depth=2, heads=8, mlp_dim=128, dropout=0.1,
                    emb_dropout=0.1)
    elif args.model == "cvit":
        model = CrossViT(image_size=32, num_classes=10, sm_dim=64,
                         lg_dim=128, sm_patch_size=8, sm_enc_depth=2,
                         sm_enc_heads=8, sm_enc_mlp_dim=128,
                         sm_enc_dim_head=64, lg_patch_size=16,
                         lg_enc_depth=2, lg_enc_heads=8,
                         lg_enc_mlp_dim=128, lg_enc_dim_head=64,
                         cross_attn_depth=2, cross_attn_heads=8,
                         cross_attn_dim_head=64, depth=3, dropout=0.1,
                         emb_dropout=0.1)
    else:
        print("not a valid model entered in the arguments. Please select between: {r18, vit, cvit}")
        return

    # Define the loss
    criterion = nn.CrossEntropyLoss(reduction="sum")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                            weight_decay=args.weight_decay)

    if use_saved_models:
        with open(args.model + " trained model.pkl", "rb") as f:
            model = pickle.load(f)
        run_test(model, device, testloader, criterion)
        return

    for epoch in range(1, args.epochs + 1):
        train(model, trainloader, optimizer, criterion, device, epoch)
        run_test(model, device, valloader, criterion, set="Validation")
        adjust_learning_rate(optimizer, epoch - 1, args)

    # plotting the validation accuracies progress
    x_data = np.arange(1, args.epochs + 1, dtype=int)
    y_data = np.array(validation_set_accuracies)
    plt.figure(figsize=(8, 4))
    plt.plot(x_data, y_data, label='training progress', color='blue')
    plt.title('Line Plot of validation set accuracies in each epoch')
    plt.xlabel('epochs')
    plt.ylabel('validation set accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(args.model + " validation set accuracy progress.png")

    # saving the model
    if args.save_model:
        with open(args.model + " trained model.pkl", "wb") as f:
            pickle.dump(model, f)

    run_test(model, device, testloader, criterion)


if __name__ == '__main__':
    args = parse_args()
    run(args, False)


# Training setups from ChatGPT
# https://chatgpt.com/share/691c4f67-4094-800c-980b-0770d44a8329

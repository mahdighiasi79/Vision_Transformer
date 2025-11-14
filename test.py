import torch


if __name__ == "__main__":
    a = torch.randn(2, 3, 4)
    print(a[:, 0, :])

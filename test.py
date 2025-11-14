import torch


if __name__ == "__main__":
    a = torch.randn(2, 3, 4)
    b = torch.randn(4, 4)
    d = torch.randn(2)
    c = torch.einsum("bnd,dd->bnd", a, b)
    print(c)

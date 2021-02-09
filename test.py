import torch
import torch.fft as fft

if __name__ == '__main__':
    a = torch.randn(5,5)
    a_ = fft.fftn(a, dim=[0,1])
    print(a_.size())
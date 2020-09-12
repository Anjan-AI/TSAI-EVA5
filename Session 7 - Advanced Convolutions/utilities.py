import torch 
from torchsummary import summary

def set_cuda():
    use_cuda = torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


def print_model_summary(model_class, input_size):
    device = set_cuda()
    print(device)
    model = model_class.to(device)
    print(summary(model, input_size=input_size))
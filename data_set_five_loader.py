from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

setting = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5,))
])
all_data = datasets.MNIST(root = '/data', train = True, download = True, transform = setting)
five = (all_data.targets == 5).nonzero(as_tuple=True)[0]
five_data = DataLoader(Subset(all_data, five), batch_size = 128, shuffle = True)

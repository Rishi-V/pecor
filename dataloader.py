import torch
from torchvision import datasets
from torchvision import transforms
import torch.utils.data
class perchDataloader(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __init__(self,root,transform):
        super(perchDataloader,self).__init__(root,transform)
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(perchDataloader, self).__getitem__(index)
        #hard code the gt to be the first one
        if index == 0:
            return (original_tuple[0],-1)
        return (original_tuple[0],[index-1])

# EXAMPLE USAGE:
# instantiate the dataset and dataloader
class outputData:
    def loadedData(data_dir):
        dataset = perchDataloader(data_dir, transform=transforms.ToTensor()) # our custom dataset
        dataloader = torch.utils.data.DataLoader(dataset)
        return dataloader
    

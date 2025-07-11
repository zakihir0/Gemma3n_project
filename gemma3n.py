#%%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%%
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import random_split, SubsetRandomSampler
from torchvision import datasets, transforms, models 
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models

dir0='/kaggle/input/mushroom1/merged_dataset'

classes=[]
paths=[]
for dirname, _, filenames in os.walk(dir0):
    for filename in filenames:
        classes+=[dirname.split('/')[-1]]
        paths+=[(os.path.join(dirname, filename))]   

dataset0=datasets.ImageFolder(root=dir0)
class_names=dataset0.classes
print(class_names)
print(len(class_names))

N=list(range(len(classes)))
normal_mapping=dict(zip(class_names,N)) 
reverse_mapping=dict(zip(N,class_names))            

data=pd.DataFrame(columns=['path','class','label'])
data['path']=paths
data['class']=classes
data['label']=data['class'].map(normal_mapping)
print(len(data))

#%%
def create_path_label_list(df):
    path_label_list = []
    for _, row in df.iterrows():
        path = row['path']
        label = row['label']
        path_label_list.append((path, label))
    return path_label_list

path_label = create_path_label_list(data)
path_label = random.sample(path_label,20000)
print(len(path_label))
print(path_label[0:3])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path_label, transform=None):
        self.path_label = path_label
        self.transform = transform

    def __len__(self):
        return len(self.path_label)

    def __getitem__(self, idx):
        path, label = self.path_label[idx]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    

class DataModule(LightningDataModule):

    def __init__(self, 
                 data_source=None, 
                 path_label=None, 
                 root_dir=None,
                 batch_size=32, 
                 train_split=0.8,
                 custom_transform=None):

        super().__init__()
        
        # Auto-detect data source if not specified
        if data_source is None:
            if path_label is not None:
                data_source = 'custom'
            elif root_dir is not None:
                data_source = 'imagefolder'
        
        self.data_source = data_source
        self.path_label = path_label
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_split = train_split
        
        # Set up transforms
        if custom_transform is not None:
            self.transform = custom_transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224),             # resize shortest side to 224 pixels
                transforms.CenterCrop(224),         # crop longest side to 224 pixels at center            
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])
            ])
        
        # Initialize dataset containers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage=None):
        """Setup datasets based on the data source type."""
        
        if self.data_source == 'custom' and self.path_label is not None:
            # Use custom dataset approach
            dataset = CustomDataset(self.path_label, self.transform)
            dataset_size = len(dataset)
            train_size = int(self.train_split * dataset_size) 
            val_size = dataset_size - train_size
            
            print(f"Custom dataset - Train size: {train_size}, Val size: {val_size}")
            
            self.train_dataset = torch.utils.data.Subset(dataset, range(train_size))
            self.val_dataset = torch.utils.data.Subset(dataset, range(train_size, dataset_size))
            
        elif self.data_source == 'imagefolder' and self.root_dir is not None:
            # Use ImageFolder approach
            dataset = datasets.ImageFolder(root=self.root_dir, transform=self.transform)
            n_data = len(dataset)
            n_train = int(self.train_split * n_data)
            n_val = n_data - n_train
            
            print(f"ImageFolder dataset - Train size: {n_train}, Val size: {n_val}")
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [n_train, n_val]
            )
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            
        else:
            raise ValueError(
                "Must specify either 'custom' data_source with path_label or "
                "'imagefolder' data_source with root_dir"
            )
    
    def train_dataloader(self):
        """Return training data loader."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Return validation data loader."""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """Return test data loader (alias for val_dataloader for compatibility)."""
        return self.val_dataloader()
    
    def __len__(self):
        """Return total dataset size."""
        total_size = 0
        if self.train_dataset is not None:
            total_size += len(self.train_dataset)
        if self.val_dataset is not None:
            total_size += len(self.val_dataset)
        return total_size
    
    def get_num_classes(self):
        """Get number of classes in the dataset."""
        if self.data_source == 'imagefolder' and self.root_dir is not None:
            temp_dataset = datasets.ImageFolder(root=self.root_dir)
            return len(temp_dataset.classes)
        elif self.data_source == 'custom' and self.path_label is not None:
            # Assuming CustomDataset has a way to get number of classes
            temp_dataset = CustomDataset(self.path_label, None)
            if hasattr(temp_dataset, 'num_classes'):
                return temp_dataset.num_classes
            else:
                # If CustomDataset doesn't have num_classes, you'll need to implement this
                print("Warning: Cannot determine number of classes for custom dataset")
                return None
        return None

import timm

class ConvolutionalNetwork(LightningModule):

    def __init__(self, num_classes):
        super().__init__()
        self.base_model = timm.create_model('resnet152', 
                                            pretrained=True, 
                                            num_classes=len(class_names))

    def forward(self, x):
        return self.base_model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("test_loss", loss)
        self.log("test_acc", acc)
    

if __name__ == '__main__':
    datamodule = DataModule(path_label=path_label)
    datamodule.setup()
    
    model = ConvolutionalNetwork(num_classes=len(class_names))

    trainer = L.Trainer(
        max_epochs=4, 
        accelerator="cpu", 
        devices=1) 
    
    trainer.fit(model, datamodule)
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()
    trainer.test(dataloaders=test_loader)

for images, labels in datamodule.test_dataloader():
    break
im=make_grid(images,nrow=8)

plt.figure(figsize=(12,12))
plt.imshow(np.transpose(im.numpy(),(1,2,0)))

inv_normalize=transforms.Normalize(mean=[-0.485/0.229,-0.456/0.224,-0.406/0.225],
                                   std=[1/0.229,1/0.224,1/0.225])
im=inv_normalize(im)

plt.figure(figsize=(12,12))
plt.imshow(np.transpose(im.numpy(),(1,2,0)))

device = torch.device("cpu")   #"cuda:0"

model.eval()
y_true=[]
y_pred=[]
with torch.no_grad():
    for test_data in datamodule.test_dataloader():
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        pred = model(test_images).argmax(dim=1)
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())
            y_pred.append(pred[i].item())

print(classification_report(y_true,y_pred,target_names=class_names,digits=4))

#%%
%%time
!pip install --quiet timm --upgrade 2> /dev/null
!pip install --quiet accelerate     2> /dev/null
!pip install --quiet git+https://github.com/huggingface/transformers.git 2> /dev/null

#%%
%%time
# Source: https://www.kaggle.com/models/google/gemma-3n
import kagglehub
from transformers import AutoProcessor, AutoModelForImageTextToText

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e4b")  # = 4 billion parameters
# GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b")  # = 2 billion parameters
processor  = AutoProcessor.from_pretrained(GEMMA_PATH)
model      = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")

def ask_gemma(prompt):
    input_ids = processor(text=str(prompt), return_tensors="pt").to(model.device, dtype=model.dtype)
    outputs   = model.generate(**input_ids, max_new_tokens=512, disable_compile=True)
    text = processor.batch_decode(
        outputs,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )
    return text[0]  # text[0] injects <bos>prompt

def ask_gemma_with_image(image_path, prompt="What do you see in this image?"):
    """
    Process image with text prompt using Gemma3n
    
    Args:
        image_path: Path to the image file
        prompt: Text prompt to ask about the image
    
    Returns:
        Generated text response
    """
    from PIL import Image
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    
    # Process both image and text
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    # Generate response
    outputs = model.generate(**inputs, max_new_tokens=512, disable_compile=True)
    
    # Decode response
    response = processor.batch_decode(
        outputs,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )
    return response[0]

def classify_mushroom_with_gemma(image_path, class_names):
    """
    Classify mushroom image using Gemma3n with structured prompt
    
    Args:
        image_path: Path to mushroom image
        class_names: List of possible mushroom classes
    
    Returns:
        Classification result
    """
    prompt = f"""Look at this mushroom image. 
What type of mushroom is this? 
Choose from these options: {', '.join(class_names)}
Provide a clear answer with reasoning."""
    
    return ask_gemma_with_image(image_path, prompt)

def analyze_mushroom_features(image_path):
    """
    Analyze detailed features of a mushroom image
    """
    prompt = """Analyze this mushroom image in detail. Describe:
1. Cap shape and color
2. Stem characteristics  
3. Gill/pore structure
4. Overall size and appearance
5. Any distinctive features"""
    
    return ask_gemma_with_image(image_path, prompt)

def test_gemma_with_mushroom_images(num_samples=3):
    """
    Test Gemma3n with random mushroom images from the dataset
    """
    import random
    
    print("Testing Gemma3n with mushroom images...")
    print("="*50)
    
    # Get random samples from the dataset
    random_samples = random.sample(path_label, min(num_samples, len(path_label)))
    
    for i, (image_path, label_idx) in enumerate(random_samples):
        print(f"\n--- Sample {i+1} ---")
        print(f"Image path: {image_path}")
        print(f"True class: {class_names[label_idx]}")
        
        try:
            # Basic image description
            print("\n1. Basic image description:")
            description = ask_gemma_with_image(image_path, "Describe this image in detail.")
            print(description)
            
            # Mushroom classification
            print("\n2. Mushroom classification:")
            classification = classify_mushroom_with_gemma(image_path, class_names)
            print(classification)
            
            # Feature analysis
            print("\n3. Feature analysis:")
            analysis = analyze_mushroom_features(image_path)
            print(analysis)
            
        except Exception as e:
            print(f"Error processing image: {e}")
        
        print("\n" + "="*50)

def compare_resnet_vs_gemma(image_path, label_idx):
    """
    Compare ResNet152 prediction vs Gemma3n analysis for the same image
    """
    print(f"Comparing ResNet152 vs Gemma3n for image: {image_path}")
    print(f"True class: {class_names[label_idx]}")
    print("-"*40)
    
    # ResNet152 prediction (would need model to be loaded)
    print("ResNet152 prediction: [Model needs to be loaded]")
    
    # Gemma3n analysis
    print("\nGemma3n analysis:")
    try:
        classification = classify_mushroom_with_gemma(image_path, class_names)
        print(classification)
    except Exception as e:
        print(f"Error: {e}")

# %%
%%time
prompt_history = ask_gemma("It was a dark and stormy night.")
print(prompt_history)
print()

# %%
# Example usage (uncomment to run):
# test_gemma_with_mushroom_images(num_samples=2)
# 
# # Test with specific image:
# if path_label:
#     sample_path, sample_label = path_label[0]
#     result = classify_mushroom_with_gemma(sample_path, class_names)
#     print(f"Classification result: {result}")

# %%

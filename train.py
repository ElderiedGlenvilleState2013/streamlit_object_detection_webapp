
import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model.model import BBoxClassifierCNN

class BBoxDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Read labels and ensure class_idx present
        self.df = pd.read_csv(csv_file)
        if 'class_idx' not in self.df.columns:
            raise ValueError("labels.csv must include a 'class_idx' column")
        # Fill missing class labels with 0 (background)
        self.df['class_idx'] = self.df['class_idx'].fillna(0).astype(int)

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        img = Image.open(img_path).convert('L')

        # bbox: normalized [cx, cy, width, height]
        bbox = torch.tensor(
            row[['x_center','y_center','width','height']].astype(float).values,
            dtype=torch.float32
        )
        # class label
        cls = torch.tensor(row['class_idx'], dtype=torch.long)

        if self.transform:
            img = self.transform(img)
        return img, bbox, cls


def train_epoch(model, loader, reg_criterion, cls_criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, bboxes, labels in loader:
        imgs, bboxes, labels = imgs.to(device), bboxes.to(device), labels.to(device)
        optimizer.zero_grad()
        bbox_out, cls_logits = model(imgs)
        loss_reg = reg_criterion(bbox_out, bboxes)
        loss_cls = cls_criterion(cls_logits, labels)
        loss = loss_reg + loss_cls
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def main():
    img_size, batch_size, lr, epochs = 256, 16, 1e-3, 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = BBoxDataset(
        csv_file='data/labels.csv',
        img_dir='data/images',
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BBoxClassifierCNN(img_size=img_size, num_classes=2).to(device)
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        loss = train_epoch(model, loader, reg_criterion, cls_criterion, optimizer, device)
        print(f"Epoch {epoch}/{epochs} — Loss: {loss:.4f}")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/bbox_cnn_cls.pth')
    print('Saved checkpoint to checkpoints/bbox_cnn_cls.pth')

if __name__ == '__main__':
    main()


###############  CODE FOR CUST MODEL SINGL PARAM
# import os
# import pandas as pd
# from PIL import Image

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms

# from model.model import BBoxClassifierCNN

# class BBoxDataset(Dataset):
#     def __init__(self, csv_file, img_dir, transform=None):
#         self.df = pd.read_csv(csv_file)
#         self.img_dir = img_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         img_path = os.path.join(self.img_dir, row['filename'])
#         img = Image.open(img_path).convert('L')

#         # normalized center/wh and class index
#         bbox = torch.tensor(row[['x_center','y_center','width','height']].values.astype('float32'))
#         cls  = torch.tensor(int(row['class_idx']), dtype=torch.long)

#         if self.transform:
#             img = self.transform(img)
#         return img, bbox, cls

# def train_epoch(model, loader, reg_loss, cls_loss, opt, device):
#     model.train()
#     tot = 0.0
#     for imgs, bboxes, labels in loader:
#         imgs, bboxes, labels = imgs.to(device), bboxes.to(device), labels.to(device)
#         opt.zero_grad()
#         bbox_out, cls_logits = model(imgs)
#         loss = reg_loss(bbox_out, bboxes) + cls_loss(cls_logits, labels)
#         loss.backward()
#         opt.step()
#         tot += loss.item()
#     return tot/len(loader)

# def main():
#     img_size, bs, lr, epochs = 256, 16, 1e-3, 20
#     dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     tf = transforms.Compose([
#         transforms.Resize((img_size,img_size)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,),(0.5,)),
#     ])
#     ds = BBoxDataset('data/labels.csv','data/images',transform=tf)
#     lo = DataLoader(ds, batch_size=bs, shuffle=True)

#     model = BBoxClassifierCNN(img_size=img_size, num_classes=2).to(dev)
#     reg_loss = nn.MSELoss()
#     cls_loss = nn.CrossEntropyLoss()
#     opt = optim.Adam(model.parameters(), lr=lr)

#     for e in range(1, epochs+1):
#         loss = train_epoch(model, lo, reg_loss, cls_loss, opt, dev)
#         print(f'Epoch {e}/{epochs} — Loss: {loss:.4f}')

#     os.makedirs('checkpoints', exist_ok=True)
#     torch.save(model.state_dict(), 'checkpoints/bbox_cnn_cls.pth')
#     print('Saved checkpoint to checkpoints/bbox_cnn_cls.pth')

# if __name__=='__main__':
#     main()



########## This the OG Code #############

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import pandas as pd
# import os

# from model.model import BBoxCNN

# # ---------------- Dataset ---------------- #
# class BBoxDataset(Dataset):
#     def __init__(self, csv_file, img_dir, transform=None):
#         self.labels = pd.read_csv(csv_file)
#         self.img_dir = img_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
#         image = Image.open(img_path).convert("L")  # CHANGED: convert to grayscale
#         bbox = self.labels.iloc[idx, 1:].values.astype("float32")  # [x_center, y_center, w, h]

#         if self.transform:
#             image = self.transform(image)

#         bbox = torch.tensor(bbox, dtype=torch.float32)
#         return image, bbox

# # ---------------- Training ---------------- #
# def train(model, dataloader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0.0
#     for images, targets in dataloader:
#         images, targets = images.to(device), targets.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     return total_loss / len(dataloader)

# # ---------------- Main ---------------- #
# def main():
#     # Hyperparameters
#     batch_size = 16
#     lr = 0.001
#     num_epochs = 20

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Dataset & Loader
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         # Optional: Normalize grayscale channel
#         transforms.Normalize((0.5,), (0.5,))  # CHANGED: normalize 1 channel
#     ])

#     dataset = BBoxDataset(
#         csv_file="data/labels.csv",
#         img_dir="data/images",
#         transform=transform
#     )
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # Model
#     model = BBoxCNN().to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     # Training loop
#     for epoch in range(num_epochs):
#         loss = train(model, dataloader, criterion, optimizer, device)
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

#     # Save model
#     torch.save(model.state_dict(), "bbox_cnn.pth")
#     print("Model saved to bbox_cnn.pth")

# if __name__ == "__main__":
#     main()


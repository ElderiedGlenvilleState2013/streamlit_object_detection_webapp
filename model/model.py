import torch
import torch.nn as nn
import torch.nn.functional as F

class BBoxClassifierCNN(nn.Module):
    def __init__(self, img_size=256, num_classes=2):
        super(BBoxClassifierCNN, self).__init__()
        # Convolutional backbone (grayscale input)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # Dynamically figure n_feats
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size)
            feat  = self.pool(F.relu(self.conv1(dummy)))
            feat  = self.pool(F.relu(self.conv2(feat)))
            feat  = self.pool(F.relu(self.conv3(feat)))
            n_feats = feat.numel()

        # Shared fc
        self.fc1 = nn.Linear(n_feats, 128)
        # Heads: 4 box coords + num_classes logits
        self.bbox_head  = nn.Linear(128, 4)
        self.class_head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        bbox_logits  = self.bbox_head(x)      # raw [cx,cy,w,h]
        class_logits = self.class_head(x)     # raw class scores
        return bbox_logits, class_logits

    




class BBoxCNN(nn.Module):
    def __init__(self, img_size=256):
        super(BBoxCNN, self).__init__()
        # Convolutional backbone (grayscale input)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # Dynamically compute flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size)
            feat  = self.pool(F.relu(self.conv1(dummy)))
            feat  = self.pool(F.relu(self.conv2(feat)))
            feat  = self.pool(F.relu(self.conv3(feat)))
            n_feats = feat.numel()

        # Fully connected layers
        self.fc1 = nn.Linear(n_feats, 128)
        self.fc2 = nn.Linear(128, 4)  # (x_center, y_center, width, height)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (16, img/2, img/2)
        x = self.pool(F.relu(self.conv2(x)))  # -> (32, img/4, img/4)
        x = self.pool(F.relu(self.conv3(x)))  # -> (64, img/8, img/8)

        x = x.view(x.size(0), -1)             # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                       # Output 4 bbox coords
        return x


## OG bbox cnn model with older labels 

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class BBoxCNN(nn.Module):
#     def __init__(self, img_size=256):
#         super(BBoxCNN, self).__init__()
#         # Convolutional layers (grayscale input)
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.3)

#         # Dynamically compute flattened feature size
#         with torch.no_grad():
#             dummy = torch.zeros(1, 1, img_size, img_size)
#             feat = self.pool(F.relu(self.conv1(dummy)))
#             feat = self.pool(F.relu(self.conv2(feat)))
#             feat = self.pool(F.relu(self.conv3(feat)))
#             n_feats = feat.numel()

#         # Fully connected layers
#         self.fc1 = nn.Linear(n_feats, 128)
#         self.fc2 = nn.Linear(128, 4)  # (x, y, w, h)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))  # -> (16, img/2, img/2)
#         x = self.pool(F.relu(self.conv2(x)))  # -> (32, img/4, img/4)
#         x = self.pool(F.relu(self.conv3(x)))  # -> (64, img/8, img/8)

#         x = x.view(x.size(0), -1)             # Flatten
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)                       # Output 4 values
#         return x

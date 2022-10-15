import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(ResBlock, self).__init__()
        layer_type = "Conv2d" if mode == 'encode' else "ConvTranspose2d"
        self.conv1 = getattr(nn, layer_type)(c_in, c_out, kernel_size, stride, padding)
        self.conv2 = getattr(nn, layer_type)(c_out, c_out, 3, 1, 1)
        """
        if mode == 'encode':
            self.conv1 = nn.Conv2d(c_in, c_out, kernel_size, stride, padding)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, kernel_size, stride, padding)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
        """

    def forward(self, x):
        relu = F.leaky_relu(self.conv1(x))
        conv2 = F.leaky_relu(self.conv2(relu))
        return x + conv2


class ResAutoEncoder(nn.Module):

    def __init__(self, encoding_size=256):
        super(ResAutoEncoder, self).__init__()
        self.encoding_size = encoding_size
        self.cn1 = nn.Conv2d(1, 32, 3, 2, 1) # 32 64 64
        self.rb1 = ResBlock(32, 32, 3, 1, 1, 'encode') # 32 64 64
        self.cn2 = nn.Conv2d(32, 32, 3, 2, 1) # 32 32 32
        self.rb2 = ResBlock(32, 32, 3, 1, 1, 'encode') # 32 32 32
        self.cn3 = nn.Conv2d(32, 16, 3, 2, 1) # 16 16 16
        self.rb3 = ResBlock(16, 16, 3, 1, 1, 'encode') # 16 16 16
        self.cn4 = nn.Conv2d(16, 16, 3, 2, 1) # 16 8 8
        self.flatten = torch.nn.Flatten()
        self.dense = nn.Linear(1024, 2*encoding_size)
 
        self.d_dense = nn.Linear(encoding_size, 1024)
        self.unflatten = torch.nn.Unflatten(1, (16, 8, 8))
        self.d_cn1 = nn.ConvTranspose2d(16, 16, 2, 2, 0) # 16 16 16
        self.d_rb1 = ResBlock(16, 16, 3, 1, 1, 'decode') # 16 16 16
        self.d_cn2 = nn.ConvTranspose2d(16, 32, 2, 2, 0) # 16 32 32
        self.d_rb2 = ResBlock(32, 32, 3, 1, 1, 'decode') # 32 32 32
        self.d_cn3 = nn.ConvTranspose2d(32, 32, 2, 2, 0) # 32 64 64
        self.d_rb3 = ResBlock(32, 32, 3, 1, 1, 'decode') # 32 64 64
        self.d_cn4 = nn.ConvTranspose2d(32, 1, 2, 2, 0) # 1 128 128
    
    def encode(self, x): 
        x = F.leaky_relu(self.cn1(x))
        x = self.rb1(x)
        x = F.leaky_relu(self.cn2(x))
        x = self.rb2(x)
        x = F.leaky_relu(self.cn3(x))
        x = self.rb3(x)
        x = F.leaky_relu(self.cn4(x))
        x = self.flatten(x)
        mu, std = self.dense(x).chunks(2)

        
        
        return x

    def forward(self, x):
        x = self.encode(x)
        x = F.leaky_relu(self.d_dense(x))
        x = self.unflatten(x)
        x = F.leaky_relu(self.d_cn1(x))
        x = self.d_rb1(x)
        x = F.leaky_relu(self.d_cn2(x))
        x = self.d_rb2(x)
        x = F.leaky_relu(self.d_cn3(x))
        x = self.d_rb3(x)
        x = F.leaky_relu(self.d_cn4(x))
        return x

    def load(self, path):
        self.load_state_dict(torch.load(path))
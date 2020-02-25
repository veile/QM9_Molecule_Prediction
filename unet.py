import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class up_conv3D(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv3D,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in,ch_out,kernel_size=2,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Net(nn.Module):
    def __init__(self, N):
        super(Net, self).__init__()
        
        # Encoding
        self.conv3d_11 = nn.Conv3d( 1, N, 3)
        self.conv3d_12 = nn.Conv3d(N, N, 3)
        self.pool_1 = nn.MaxPool3d(2)
        
        self.conv3d_21 = nn.Conv3d(N, 2*N, 3)
        self.conv3d_22 = nn.Conv3d(2*N, 2*N, 3)
        self.pool_2 = nn.MaxPool3d(2)
        
        # Latent space
        self.conv3d_31 = nn.Conv3d(2*N, 4*N, 3)
        self.conv3d_32 = nn.Conv3d(4*N, 4*N, 3)

        
        #Decoding
        self.up3 = up_conv3D(4*N, 2*N)
        
        self.conv3d_23 = nn.Conv3d(4*N, 2*N, 3)
        self.conv3d_24 = nn.Conv3d(2*N, 2*N, 3)
        self.up2 = up_conv3D(2*N, N)
        
        self.conv3d_13 = nn.Conv3d(2*N, N, 3)
        self.conv3d_14 = nn.Conv3d(N, 6, 3) # HCONF + background
        
        
    def forward(self, x):
        x1 = F.relu(self.conv3d_11(x) )
        x1 = F.relu(self.conv3d_12(x1))
        
        x2 = self.pool_1(x1)
        x2 = F.relu(self.conv3d_21(x2) )
        x2 = F.relu(self.conv3d_22(x2) )
        
        x3 = self.pool_2(x2)
        x3 = F.relu(self.conv3d_31(x3) )
        x3 = F.relu(self.conv3d_32(x3) )
        
        y2 = self.up3(x3)
        ny = y2.shape[3]/2
        midx =  math.ceil( x2.shape[3]/2 )
        
        xcat = x2[:, :, (midx-math.floor(ny)):(midx+math.ceil(ny))
                      , (midx-math.floor(ny)):(midx+math.ceil(ny))
                      , (midx-math.floor(ny)):(midx+math.ceil(ny)) ]
    
        y2 = torch.cat( (xcat, y2), 1 )
  
    
        y2 = F.relu(self.conv3d_23(y2) )
        y2 = F.relu(self.conv3d_24(y2) )
    
        y1 = self.up2(y2)
        ny = y1.shape[3]/2
        midx = math.ceil( x1.shape[3]/2 )
        
        xcat = x1[:, :, (midx-math.floor(ny)):(midx+math.ceil(ny))
                      , (midx-math.floor(ny)):(midx+math.ceil(ny))
                      , (midx-math.floor(ny)):(midx+math.ceil(ny)) ]
    
        y1 = torch.cat( (xcat, y1), 1)
        y1 = F.relu(self.conv3d_13(y1) )
        y1 = F.softmax(self.conv3d_14(y1), dim=-1 )
        
        
        return y1

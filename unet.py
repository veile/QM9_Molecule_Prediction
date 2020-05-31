import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class up_conv3D(nn.Module):
    def __init__(self,ch_in,ch_out, kernel_size):
        super(up_conv3D,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in,ch_out, kernel_size=kernel_size, stride=1,
                      padding=2, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Net(nn.Module):
    """
    U-net structure with 3 levels consisting of encoding part, latent space
    and decoding part.
    
    Parameters
    ----------
    N : int
        Number of channels created after first convolution. The maximum number
        of channels is 4N.
    """
    def __init__(self, N):
        # Adjust the settings of the convolution
        kernel_size = 5
        padding = 2
        dilation = 1
        super(Net, self).__init__()
        
        # All variable name is named as operation_levelstep, so a convolution
        # in the start of the first level would be conv3d_11
        # Encoding
        self.conv3d_11 = nn.Conv3d(1, N, kernel_size, dilation=dilation)
        self.conv3d_12 = nn.Conv3d(N, N, kernel_size, dilation=dilation)
        self.pool_1 = nn.MaxPool3d(2)
        
        self.conv3d_21 = nn.Conv3d(N, 2*N, kernel_size, padding=padding,
                                   dilation=dilation)
        self.conv3d_22 = nn.Conv3d(2*N, 2*N, kernel_size, padding=padding,
                                   dilation=dilation)
        self.pool_2 = nn.MaxPool3d(2)
        
        # Latent space
        self.conv3d_31 = nn.Conv3d(2*N, 4*N, kernel_size, padding=padding,
                                   dilation=dilation)
        self.conv3d_32 = nn.Conv3d(4*N, 4*N, kernel_size, padding=padding,
                                   dilation=dilation)

        
        #Decoding
        self.up3 = up_conv3D(4*N, 2*N, kernel_size)
        
        self.conv3d_23 = nn.Conv3d(4*N, 2*N, kernel_size, padding=padding,
                                   dilation=dilation)
        self.conv3d_24 = nn.Conv3d(2*N, 2*N, kernel_size, padding=padding,
                                   dilation=dilation)
        self.up2 = up_conv3D(2*N, N, kernel_size)
        
        self.conv3d_13 = nn.Conv3d(2*N, N, kernel_size, padding=padding,
                                   dilation=dilation)
        self.conv3d_14 = nn.Conv3d(N, 6, 1) # HCONF + background
        
        
    def forward(self, x):
        # ====================================================================
        # 3 Levels neural network
        # ====================================================================
        x1 = F.relu(self.conv3d_11(x) )
        x1 = F.relu(self.conv3d_12(x1))
        
        x2 = self.pool_1(x1)
        x2 = F.relu(self.conv3d_21(x2) )
        x2 = F.relu(self.conv3d_22(x2) )
        
        x3 = self.pool_2(x2)
        x3 = F.relu(self.conv3d_31(x3) )
        x3 = F.relu(self.conv3d_32(x3) )
        
        y2 = self.up3(x3)
        
        # Concatenate the decoding part with a section from the encoding part 
        ny = y2.shape[3]/2
        midx =  math.ceil( x2.shape[3]/2 )
        
        xcat = x2[:, :, (midx-math.floor(ny)):(midx+math.ceil(ny))
                      , (midx-math.floor(ny)):(midx+math.ceil(ny))
                      , (midx-math.floor(ny)):(midx+math.ceil(ny)) ]
                      
        y2 = torch.cat( (xcat, y2), 1 )
        y2 = F.relu(self.conv3d_23(y2) )
        y2 = F.relu(self.conv3d_24(y2) )
    
    
        y1 = self.up2(y2)
        
        # Concatenate the decoding part with a section from the encoding part 
        ny = y1.shape[3]/2
        midx = math.ceil( x1.shape[3]/2 )
        
        xcat = x1[:, :, (midx-math.floor(ny)):(midx+math.ceil(ny))
                      , (midx-math.floor(ny)):(midx+math.ceil(ny))
                      , (midx-math.floor(ny)):(midx+math.ceil(ny)) ]
    
        y1 = torch.cat( (xcat, y1), 1)
        y1 = F.relu(self.conv3d_13(y1) )
        y1 =  self.conv3d_14(y1)
        
        return y1
        # ====================================================================
        # 2 Levels neural network
        # ====================================================================
        #x1 = F.relu(self.conv3d_11(x) )
        #x1 = F.relu(self.conv3d_12(x1))
        #
        #x2 = self.pool_1(x1)
        #x2 = F.relu(self.conv3d_21(x2) )
        #x2 = F.relu(self.conv3d_22(x2) )
        #
        #
        #y1 = self.up2(x2)
        #
        ## Concatenate the decoding part with a section from the encoding part 
        #ny = y1.shape[3]/2
        #midx = math.ceil( x1.shape[3]/2 )
        #
        #xcat = x1[:, :, (midx-math.floor(ny)):(midx+math.ceil(ny))
        #              , (midx-math.floor(ny)):(midx+math.ceil(ny))
        #              , (midx-math.floor(ny)):(midx+math.ceil(ny)) ]
        #
        #y1 = torch.cat( (xcat, y1), 1)
        #y1 = F.relu(self.conv3d_13(y1) )
        #y1 =  self.conv3d_14(y1)
        #
        #return y1

if __name__=="__main__":
    model = Net(8)
    print( sum(p.numel() for p in model.parameters() if p.requires_grad) )
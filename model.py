import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConvBlock(nn.Module):
    '''
    Function: ResConvBlock.__init__
    Purpose: Initializes the stored block.
    Inputs:
      ch_in  - int; the number of input channels entering the conv block.
      ch_out - int; the number of output channels produced by the conv block.
    Returns:
      N/A
    Behaviour:
      Builds a residual conv block matching the saved ResCNN from Martin's work.
      If the input and output channel counts differ, the shortcut uses a 1x1 
      convolution to match dimensions.
    '''
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
        )
        self.shortcut = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False) if ch_in != ch_out else nn.Identity()

    '''
    Function: ResConvBlock.forward
    Purpose: Runs input image features through convolution block.
    Inputs:
      x - torch.Tensor; input feature map with shape [B, ch_in, H, W].
    Returns:
      torch.Tensor; output feature map with shape [B, ch_out, H, W].
    Behaviour:
      Applies the main convolutional & then applies ReLU activation 
      to the combined result (with the shortcut output if not N/A).
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x) + self.shortcut(x))


class ResCNN(nn.Module):
    
    '''
    Function: ResCNN.__init__
    Purpose: Initializes the ResCNN model architecture from Martin's classification.
    Inputs:
      img_ch - int; the number of image input channels, with a default value of 3.
      first_layer_numKernel - int; the number of kernels in the first encoder block,
                              with a default value of 64.
    Returns:
      N/A
    Behaviour:
      Builds five-block residual CNN encoder, max-pooling layers, encoder dropout,
      learnable magnification embedding, and two classification heads. The first
      head outputs benign/malignant logits, while the second outputs subtype logits.
      (Note that for the most part the second head wasn't used for this part of the
      project.)
    '''
    def __init__(self, img_ch: int = 3, first_layer_numKernel: int = 64):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = ResConvBlock(ch_in=img_ch, ch_out=first_layer_numKernel)
        self.Conv2 = ResConvBlock(ch_in=first_layer_numKernel, ch_out=2 * first_layer_numKernel)
        self.Conv3 = ResConvBlock(ch_in=2 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)
        self.Conv4 = ResConvBlock(ch_in=4 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)
        self.Conv5 = ResConvBlock(ch_in=8 * first_layer_numKernel, ch_out=16 * first_layer_numKernel)
        self.dropout_encoder = nn.Dropout2d(p=0.1)

        self.magn_embedding = nn.Embedding(4, 16 * first_layer_numKernel)

        self.head1 = nn.Sequential(
            nn.Linear(16 * first_layer_numKernel, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )
        self.head2 = nn.Sequential(
            nn.Linear(16 * first_layer_numKernel, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 8),
        )
    
    '''
    Function: ResCNN.forward
    Purpose: Forward pass through the ResCNN model.
    Inputs:
      x    - torch.Tensor; batch of input images with shape [B, 3, H, W].
      magn - torch.Tensor; batch of magnification tokens with shape [B].
    Returns:
      tuple[torch.Tensor, torch.Tensor]; the first tensor with binary
      benign/malignant logits with shape [B, 2], and the second tensor with
      subtype logits with shape [B, 8].
    Behaviour:
      Passes the image batch through the residual encoder, does the dropout, adds
      the learned magnification token to the final feature map, global averages
      the spatial dimensions, and sends the resulting feature vector through both
      classification heads. Tada : )
    '''
    def forward(self, x: torch.Tensor, magn: torch.Tensor):
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))
        x5 = self.dropout_encoder(x5)

        magn_token = self.magn_embedding(magn.long()).unsqueeze(-1).unsqueeze(-1)
        x5 = x5 + magn_token
        x_feat = torch.mean(x5, dim=[2, 3])
        return self.head1(x_feat), self.head2(x_feat)

'''
Function: load_rescnn
Purpose: Loads the trained ResCNN.
Inputs:
  weights_path - str; path to the saved model checkpoint weights.
  device       - str | torch.device | None; device used to load and run the model.
                 If None, CUDA is used when available, otherwise CPU is used.
Returns:
  ResCNN; a ResCNN model with checkpoint weights loaded & set to evaluation mode.
Behaviour:
  Creates a ResCNN instance, selects the target device, loads the saved state
  dictionary, copies the weights into the model, moves the model to the device,
  switches it to eval mode, and gives you a perfectly usable model courtesy of 
  Martin.
'''
def load_rescnn(weights_path: str, device: str | torch.device | None = None) -> ResCNN:
    """Create a ResCNN instance, load checkpoint weights, move to device, and set eval mode."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = ResCNN()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

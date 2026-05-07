import torch
import torch.nn as nn
from timm import create_model

# NOTE: all file paths in this module assume Google Colab with Google Drive mounted.
# Update paths to match your local directory structure if running outside Colab.
# Checkpoint path example: /content/drive/MyDrive/vit_breakhis_best.pth

class ViT_BreaKHis(nn.Module):
    '''
    Class: ViT_BreaKHis
    Purpose: Vision Transformer (ViT-Small) with learnable magnification token
             embedding and dual MLP heads for binary and subtype classification
             on the BreaKHis dataset.
    '''
    def __init__(
        self,
        model_name='vit_small_patch16_224',
        pretrained=True,
        num_classes_binary=2,
        num_classes_subtype=8,
        num_magnifications=4,
        dropout=0.2
    ):
        '''
        Function: ViT_BreaKHis.__init__
        Purpose: Initializes ViT encoder, magnification embedding, and dual
                 classification heads.
        Inputs:
          model_name          - str; timm model identifier.
                                Defaults to vit_small_patch16_224.
          pretrained          - bool; load ImageNet-1K weights. Defaults to True.
          num_classes_binary  - int; number of binary output classes. Defaults to 2.
          num_classes_subtype - int; number of subtype output classes. Defaults to 8.
          num_magnifications  - int; number of magnification levels
                                (40x/100x/200x/400x). Defaults to 4.
          dropout             - float; dropout probability. Defaults to 0.2.
        Returns:
          N/A
        Behaviour:
          Loads the pretrained ViT encoder from timm with the classification head
          removed. Sets the embedding dimension based on the model variant.
          Builds a learnable magnification embedding, dropout layer, and two
          MLP classification heads for binary and subtype prediction.
        '''
        super(ViT_BreaKHis, self).__init__()

        # load pretrained ViT encoder, remove classification head
        self.vit_encoder = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='token'
        )

        # embedding dimension: vit_tiny=192, vit_small=384, vit_base=768
        if 'tiny' in model_name:
            self.embed_dim = 192
        elif 'small' in model_name:
            self.embed_dim = 384
        elif 'base' in model_name:
            self.embed_dim = 768
        else:
            self.embed_dim = 384

        # learnable magnification token, additive fusion matching ResNet baseline
        self.magnification_embedding = nn.Embedding(
            num_embeddings=num_magnifications,
            embedding_dim=self.embed_dim
        )

        self.dropout = nn.Dropout(dropout)

        # head 1: binary benign/malignant classification
        self.binary_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes_binary)
        )

        # head 2: 8-way subtype classification
        self.subtype_head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes_subtype)
        )

    def forward(self, image, magnification):
        '''
        Function: ViT_BreaKHis.forward
        Purpose: Forward pass through ViT encoder, magnification fusion, and heads.
        Inputs:
          image         - torch.Tensor [B, 3, 224, 224]; batch of input images.
          magnification - torch.Tensor [B]; magnification tokens.
                          Encoding: 40x=0, 100x=1, 200x=2, 400x=3.
        Returns:
          tuple[torch.Tensor, torch.Tensor];
            binary logits [B, 2] and subtype logits [B, 8].
        Behaviour:
          Passes images through the ViT encoder to extract CLS token features.
          Adds the learnable magnification embedding element-wise to the image
          features before applying dropout and passing through both heads.
        '''
        # extract CLS token features from ViT encoder
        image_features = self.vit_encoder(image)

        # get magnification embedding and fuse additively
        mag_token = self.magnification_embedding(magnification)
        fused = self.dropout(image_features + mag_token)

        return self.binary_head(fused), self.subtype_head(fused)


def load_vit(weights_path, device=None):
    '''
    Function: load_vit
    Purpose: Loads a trained ViT_BreaKHis model from a saved checkpoint.
    Inputs:
      weights_path - str; path to the saved .pth checkpoint file.
                     e.g. /content/drive/MyDrive/vit_breakhis_best.pth
      device       - str | torch.device | None; defaults to CUDA if available,
                     otherwise CPU.
    Returns:
      ViT_BreaKHis; model with weights loaded and set to eval mode.
    Behaviour:
      Creates a ViT_BreaKHis instance with pretrained=False, loads the saved
      state dictionary from the checkpoint file, handles both raw state dicts
      and full checkpoint dicts containing model_state_dict, moves the model
      to the target device, and sets it to evaluation mode.
    '''
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model = ViT_BreaKHis(pretrained=False)
    checkpoint = torch.load(weights_path, map_location=device)

    # handle both raw state dict and full checkpoint dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model

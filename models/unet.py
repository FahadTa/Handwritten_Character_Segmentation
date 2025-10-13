"""
U-Net architecture for handwritten character segmentation.
Optimized for handling variability in handwriting styles, touching characters,
irregular spacing, and ink artifacts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ConvBlock(nn.Module):
    """
    Convolutional block with two conv layers, batch norm, and activation.
    Designed to capture handwriting stroke patterns and variations.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        use_dropout: bool = False,
        dropout_rate: float = 0.3,
        activation: str = 'relu'
    ):
        """
        Initialize convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_batch_norm: Whether to use batch normalization
            use_dropout: Whether to use dropout
            dropout_rate: Dropout probability
            activation: Activation function ('relu', 'leaky_relu', 'elu')
        """
        super(ConvBlock, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=not use_batch_norm
        )
        
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=not use_batch_norm
        )
        
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        
        if use_dropout:
            self.dropout = nn.Dropout2d(dropout_rate)
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through conv block.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, out_channels, H, W)
        """
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.activation(x)
        
        if self.use_dropout:
            x = self.dropout(x)
        
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.activation(x)
        
        return x


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections.
    Critical for handwritten text: helps focus on character boundaries
    and suppress background/noise in feature maps.
    """
    
    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: Optional[int] = None
    ):
        """
        Initialize attention gate.
        
        Args:
            gate_channels: Channels from decoder (gating signal)
            skip_channels: Channels from encoder (skip connection)
            inter_channels: Intermediate channels (default: skip_channels // 2)
        """
        super(AttentionGate, self).__init__()
        
        if inter_channels is None:
            inter_channels = skip_channels // 2
        
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(
        self,
        gate: torch.Tensor,
        skip: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention gating.
        
        Args:
            gate: Gating signal from decoder (B, gate_channels, H, W)
            skip: Skip connection from encoder (B, skip_channels, H', W')
            
        Returns:
            Attention-weighted skip connection
        """
        gate_feat = self.W_gate(gate)
        skip_feat = self.W_skip(skip)
        
        if gate_feat.shape[2:] != skip_feat.shape[2:]:
            gate_feat = F.interpolate(
                gate_feat,
                size=skip_feat.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        
        attention = self.relu(gate_feat + skip_feat)
        attention = self.psi(attention)
        
        return skip * attention


class Encoder(nn.Module):
    """
    U-Net encoder with progressive downsampling.
    Captures hierarchical features from fine handwriting strokes to
    global document structure.
    """
    
    def __init__(
        self,
        in_channels: int,
        encoder_channels: List[int],
        use_batch_norm: bool = True,
        use_dropout: bool = True,
        dropout_rate: float = 0.3,
        activation: str = 'relu'
    ):
        """
        Initialize encoder.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            encoder_channels: List of channel sizes for each encoder level
            use_batch_norm: Whether to use batch normalization
            use_dropout: Whether to use dropout
            dropout_rate: Dropout probability
            activation: Activation function
        """
        super(Encoder, self).__init__()
        
        self.encoder_channels = encoder_channels
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        current_channels = in_channels
        for out_channels in encoder_channels:
            self.encoder_blocks.append(
                ConvBlock(
                    current_channels,
                    out_channels,
                    use_batch_norm=use_batch_norm,
                    use_dropout=use_dropout,
                    dropout_rate=dropout_rate,
                    activation=activation
                )
            )
            current_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            List of feature maps from each encoder level
        """
        skip_connections = []
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        return skip_connections


class Decoder(nn.Module):
    """
    U-Net decoder with progressive upsampling and skip connections.
    Reconstructs high-resolution character segmentation masks with
    attention gates for better boundary localization.
    """
    
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        use_attention: bool = True,
        use_batch_norm: bool = True,
        use_dropout: bool = True,
        dropout_rate: float = 0.3,
        activation: str = 'relu'
    ):
        """
        Initialize decoder.
        
        Args:
            encoder_channels: Channel sizes from encoder
            decoder_channels: Channel sizes for decoder levels
            use_attention: Whether to use attention gates
            use_batch_norm: Whether to use batch normalization
            use_dropout: Whether to use dropout
            dropout_rate: Dropout probability
            activation: Activation function
        """
        super(Decoder, self).__init__()
        
        self.decoder_channels = decoder_channels
        self.use_attention = use_attention
        
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        if use_attention:
            self.attention_gates = nn.ModuleList()
        
        bottleneck_channels = encoder_channels[-1]
        
        for i, out_channels in enumerate(decoder_channels):
            skip_channels = encoder_channels[-(i + 2)]
            
            self.upsamples.append(
                nn.ConvTranspose2d(
                    bottleneck_channels,
                    out_channels,
                    kernel_size=2,
                    stride=2
                )
            )
            
            if use_attention:
                self.attention_gates.append(
                    AttentionGate(
                        gate_channels=out_channels,
                        skip_channels=skip_channels
                    )
                )
            
            self.decoder_blocks.append(
                ConvBlock(
                    out_channels + skip_channels,
                    out_channels,
                    use_batch_norm=use_batch_norm,
                    use_dropout=use_dropout,
                    dropout_rate=dropout_rate,
                    activation=activation
                )
            )
            
            bottleneck_channels = out_channels
    
    def forward(
        self,
        skip_connections: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            skip_connections: Feature maps from encoder
            
        Returns:
            Decoded feature map
        """
        x = skip_connections[-1]
        
        for i in range(len(self.decoder_blocks)):
            x = self.upsamples[i](x)
            
            skip = skip_connections[-(i + 2)]
            
            if self.use_attention:
                skip = self.attention_gates[i](gate=x, skip=skip)
            
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )
            
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_blocks[i](x)
        
        return x


class UNet(nn.Module):
    """
    U-Net for handwritten character instance segmentation.
    
    Architecture features for handwritten text:
    - Deep encoder for capturing handwriting style variations
    - Attention gates for precise character boundary detection
    - Skip connections preserving fine stroke details
    - Dropout for handling diverse handwriting patterns
    - Multi-scale feature processing for variable character sizes
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 64,
        encoder_channels: List[int] = [64, 128, 256, 512, 1024],
        decoder_channels: List[int] = [512, 256, 128, 64],
        use_attention: bool = True,
        use_batch_norm: bool = True,
        use_dropout: bool = True,
        dropout_rate: float = 0.3,
        activation: str = 'relu'
    ):
        """
        Initialize U-Net model.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of character classes (including background)
            encoder_channels: Channel sizes for encoder levels
            decoder_channels: Channel sizes for decoder levels
            use_attention: Use attention gates in skip connections
            use_batch_norm: Use batch normalization
            use_dropout: Use dropout for regularization
            dropout_rate: Dropout probability
            activation: Activation function type
        """
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.encoder = Encoder(
            in_channels=in_channels,
            encoder_channels=encoder_channels,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        self.decoder = Decoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            use_attention=use_attention,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        self.final_conv = nn.Conv2d(
            decoder_channels[-1],
            num_classes,
            kernel_size=1
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            Character segmentation logits (B, num_classes, H, W)
        """
        skip_connections = self.encoder(x)
        
        x = self.decoder(skip_connections)
        
        x = self.final_conv(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict segmentation masks with softmax activation.
        
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            Predicted class probabilities (B, num_classes, H, W)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class indices for each pixel.
        
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            Predicted class indices (B, H, W)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


def create_unet(config: dict) -> UNet:
    """
    Factory function to create U-Net from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured UNet model
    """
    model_config = config.get('model', {}).get('unet', {})
    
    model = UNet(
        in_channels=3,
        num_classes=model_config.get('num_classes', 64),
        encoder_channels=model_config.get('encoder_channels', [64, 128, 256, 512, 1024]),
        decoder_channels=model_config.get('decoder_channels', [512, 256, 128, 64]),
        use_attention=True,
        use_batch_norm=model_config.get('use_batch_norm', True),
        use_dropout=model_config.get('use_dropout', True),
        dropout_rate=model_config.get('dropout_rate', 0.3),
        activation=model_config.get('activation', 'relu')
    )
    
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Count total trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
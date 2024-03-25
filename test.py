import torch
import torch.nn as nn

# 输入图像大小 3x64x64
input_image = torch.randn(1, 3, 64, 64)

# 创建一个 Conv2d 层
conv_layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2)

# 对输入图像进行卷积
output_feature_map = conv_layer(input_image)
print(input_image)
print(output_feature_map)
# 输出特征图的大小
print("Output shape:", output_feature_map.shape)
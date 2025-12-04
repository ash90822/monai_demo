from monai.networks.nets import UNet
from configs.config import CFG

def get_basic_unet():
    model = UNet(
        spatial_dims=2,
        in_channels=CFG.model.in_channels,   # 單通道輸入（我們回傳 (1,H,W)）
        out_channels=CFG.model.out_channels,  # 背景 + 兩個器官
        channels=CFG.model.channels,
        strides=CFG.model.strides,
        num_res_units=CFG.model.num_res_units,
    )
    return model

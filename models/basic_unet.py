from monai.networks.nets import DynUNet
from configs.config import CFG

def get_basic_nnUnet():
    num_levels = len(CFG.model.channels)
    kernel_size = [[3, 3] for _ in range(num_levels)]
    strides = [[1, 1]] + [[s, s] for s in CFG.model.strides]
    upsample_kernel_size = [[s, s] for s in CFG.model.strides]

    model = DynUNet(
        spatial_dims=2,
        in_channels=CFG.model.in_channels,
        out_channels=CFG.model.out_channels,
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=upsample_kernel_size,
        norm_name="instance",
        deep_supervision=False,
    )
    return model
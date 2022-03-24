import segmentation_models_pytorch as smp


def setup_model(cfg, n_in, n_out):
    return smp.Unet(encoder_name="efficientnet-b4", in_channels=n_in, classes=n_out, encoder_weights="imagenet")

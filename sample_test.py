import torch
from torchvision.utils import make_grid, save_image
from diffusers import AutoencoderKL
from models.dit import MFDiT
from meanflow import MeanFlow
import os

#test

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # paths
    ckpt_path = "checkpoints/step_368000.0.pt"   # <-- change to your trained ckpt
    out_dir = "samples_test"
    os.makedirs(out_dir, exist_ok=True)

    # --- Load VAE ---
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    latent_factor = 0.18215

    # --- Build model ---
    model = MFDiT(
        input_size=32,
        patch_size=2,
        in_channels=4,
        dim=384,
        depth=8,
        num_heads=6,
        num_classes=200,
    ).to(device)

    # --- Load weights ---
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # --- Init MeanFlow ---
    meanflow = MeanFlow(
        channels=4,
        image_size=32,
        num_classes=200,
        normalizer=['mean_std', 0.0, 1/latent_factor],
        flow_ratio=0.50,
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        cfg_scale=2.0,
        cfg_uncond='u',
    )

    # --- Sampling ---
    with torch.no_grad():
        # choose classes you want to sample
        class_ids = [1, 25, 65, 123, 157]
        z = meanflow.sample_each_class(model, n_per_class=4, classes=class_ids)

        # decode to pixel space
        x = vae.decode(z).sample
        x = x * 0.5 + 0.5  # denorm to [0,1]

        # save grid
        grid = make_grid(x, nrow=len(class_ids))
        save_image(grid, os.path.join(out_dir, "sample_test.png"))

    print(f"Saved samples to {os.path.join(out_dir, 'sample_test.png')}")

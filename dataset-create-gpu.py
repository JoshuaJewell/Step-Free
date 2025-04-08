import os
import sys
from PIL import Image
from datasets import load_dataset
from diffusers.models import AutoencoderKL
from torchvision import transforms
import torch

# Initialisation
ds = load_dataset("alexandrainst/nordjylland-news-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)

# Resize function (yeah, my gpu's useless)
def custom_resize(image, max_size=2048, max_area=696320):
    width, height = image.size
    if width <= max_size and height <= max_size:
        new_width, new_height = width, height
    elif width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    new_area = new_width * new_height

    if new_area > max_area:
        # Calculate the scaling factor to reduce the area to max_area
        scale_factor = (max_area / new_area) ** 0.5
        new_width = int(new_width * scale_factor)
        new_height = int(new_height * scale_factor)

    return image.resize((new_width, new_height), Image.BILINEAR)

preprocess = transforms.Compose([
    transforms.Lambda(custom_resize),
    transforms.ToTensor(),
])

postprocess = transforms.Compose([
    transforms.ToPILImage()
])

# Dir structure
base_dir = f'{os.path.dirname(os.path.abspath(sys.argv[0]))}'
os.makedirs(base_dir, exist_ok=True)

original_dir = os.path.join(base_dir, 'Original')
vae_dir = os.path.join(base_dir, 'VAE')
os.makedirs(original_dir, exist_ok=True)
os.makedirs(vae_dir, exist_ok=True)

print(f"Original Directory: {original_dir}")
print(f"VAE Directory: {vae_dir}")

# Check VAE dir for existing files for resume
existing_files = os.listdir(vae_dir)
existing_indices = [int(f.split('.')[0]) for f in existing_files if f.endswith('.jpg')]
start_index = max(existing_indices, default=-1) + 1  # Start from the next index

# Iterate through the dataset starting from the determined index
for i in range(start_index, len(ds['train'])):
    item = ds['train'][i] 

    # Load and preprocess
    image = item['image'].convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Encode-decode
    with torch.no_grad():
        latent_dist = vae.encode(image_tensor)
        latent = latent_dist.latent_dist.sample()

    with torch.no_grad():
        decoded_image_tensor = vae.decode(latent).sample
        decoded_image_tensor = decoded_image_tensor.clamp(0, 1)

    # Postprocess from VAE
    decoded_image_tensor = decoded_image_tensor.cpu()
    decoded_image = postprocess(decoded_image_tensor.squeeze(0))

    # Postprocess from Original
    original_image = postprocess(image_tensor.cpu().squeeze(0))

    # Save
    original_image_path = os.path.join(original_dir, f'{i}.jpg')
    decoded_image_path = os.path.join(vae_dir, f'{i}.jpg')

    original_image.save(original_image_path)
    decoded_image.save(decoded_image_path)

    print(f"Decoded Image: {i}")

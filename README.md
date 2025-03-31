# Step Free Assess
This repository contains all the scripts for an AI image detection CSSL NN I am training.

Images are fed to diffusion models as latents - a distilled representation of the image that allows processing with reduced overhead. This is facilitated by a variational autoencoder (VAE), a neural network that encodes/decodes between images and latents. Because decoding by VAE is a lossy process, and a process that is currently used in all major, modern image generation implementations, it seems like an obvious candidate for detecting AI generated images.

I aim to demonstrate this effect in the following VAE's:

## Priority
These models represent a diverse range of commonly used VAE's.
[SDXL](https://huggingface.co/stabilityai/sdxl-vae): [dataset](https://huggingface.co/datasets/joshuajewell/VAEDecodedImages-SDXL) [model]()
[DALL·E](https://github.com/openai/dall-e): [dataset]() [model]()
[FLUX](https://huggingface.co/StableDiffusionVN/Flux/tree/main/Vae): [dataset]() [model]()

## Other
These models are either incredibly popular but may be similar to SDXL or are old SD models with less relevance today.
[XL_VAE_C](https://civitai.com/models/152040?modelVersionId=1023774): [dataset]() [model]()
[djMILXLVAE](https://civitai.com/models/1257377?modelVersionId=1472075): [dataset]() [model]()
[Pony Pastel](https://civitai.com/models/660613/pony-enhanced-vae-pastels): [dataset]() [model]()
[SD 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/vae): [dataset]() [model]()

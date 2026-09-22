import os
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import cv2

# Global reference to pipeline
_PIPELINE = None
_IP_ADAPTER_LOADED = False

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_pipeline():
    global _PIPELINE, _IP_ADAPTER_LOADED
    if _PIPELINE is not None:
        return _PIPELINE
    
    device = get_device()
    print(f"Loading Stable Diffusion Inpainting pipeline on {device}...")

    from diffusers import AutoPipelineForInpainting
    
    model_id = "runwayml/stable-diffusion-inpainting"
    
    if device == "cuda":
        pipe = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        pipe.to("cuda")
        try:
            pipe.enable_attention_slicing()
        except Exception as e:
            print(f"Attention slicing info: {e}")
            
        # Try loading IP-Adapter for reference-based generation
        try:
            print("Loading IP-Adapter weights for visual reference guidance...")
            pipe.load_ip_adapter(
                "h94/IP-Adapter", 
                subfolder="models", 
                weight_name="ip-adapter_sd15.bin"
            )
            _IP_ADAPTER_LOADED = True
            print("IP-Adapter successfully loaded.")
        except Exception as e:
            print(f"Warning: IP-Adapter could not be loaded immediately ({e}). Text inpainting will still work.")
            _IP_ADAPTER_LOADED = False
    else:
        pipe = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None
        )
        pipe.to("cpu")
        _IP_ADAPTER_LOADED = False

    _PIPELINE = pipe
    return _PIPELINE

def resize_for_sd(image, max_dim=768):
    """Resize image to multiples of 8 and within max_dim to conserve VRAM."""
    w, h = image.size
    scale = min(max_dim / max(w, h), 1.0)
    new_w = int((w * scale) // 8 * 8)
    new_h = int((h * scale) // 8 * 8)
    new_w = max(new_w, 256)
    new_h = max(new_h, 256)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

def inpaint_general(
    input_dict,
    prompt: str,
    negative_prompt: str = "bad quality, blurry, deformed, disfigured, distorted, lowres",
    reference_image=None,
    ref_strength: float = 0.7,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 25,
    seed: int = -1
):
    """
    Universal Inpainting function:
    - Replaces objects, adds waifu / characters, alters background / atmosphere.
    - If reference_image is provided and IP-Adapter is loaded, uses visual identity conditioning.
    """
    global _PIPELINE, _IP_ADAPTER_LOADED
    pipe = load_pipeline()
    device = get_device()

    # Extract base image and mask
    if isinstance(input_dict, dict):
        base_img = input_dict.get("background") or input_dict.get("image")
        mask_img = input_dict.get("layers", [None])[0] if "layers" in input_dict else input_dict.get("mask")
    else:
        base_img = input_dict
        mask_img = None

    if base_img is None:
        raise ValueError("Gambar dasar belum disediakan.")

    if not isinstance(base_img, Image.Image):
        base_img = Image.fromarray(base_img)
    base_img = base_img.convert("RGB")

    # If mask is None or empty, create an empty white or transparent mask
    if mask_img is None:
        # User didn't mask anything; alert or do img2img
        mask_img = Image.new("L", base_img.size, 0)
    else:
        if not isinstance(mask_img, Image.Image):
            mask_img = Image.fromarray(mask_img)
        # Convert mask to grayscale (L)
        mask_img = mask_img.convert("L")

    # Resize to SD multiples of 8
    target_size = (base_img.width, base_img.height)
    base_resized = resize_for_sd(base_img)
    mask_resized = mask_img.resize(base_resized.size, Image.Resampling.NEAREST)

    # Set up generator for reproducibility
    if seed == -1 or seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    # Configure IP-Adapter scale
    if _IP_ADAPTER_LOADED and reference_image is not None and ref_strength > 0:
        if not isinstance(reference_image, Image.Image):
            reference_image = Image.fromarray(reference_image)
        reference_image = reference_image.convert("RGB")
        pipe.set_ip_adapter_scale(ref_strength)
        ip_adapter_image = reference_image
    else:
        if _IP_ADAPTER_LOADED:
            pipe.set_ip_adapter_scale(0.0)
        ip_adapter_image = None

    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": base_resized,
        "mask_image": mask_resized,
        "guidance_scale": guidance_scale,
        "num_inference_steps": int(num_inference_steps),
        "generator": generator
    }

    if ip_adapter_image is not None and _IP_ADAPTER_LOADED:
        kwargs["ip_adapter_image"] = ip_adapter_image

    result = pipe(**kwargs).images[0]
    # Resize back to original dimensions
    result_restored = result.resize(target_size, Image.Resampling.LANCZOS)
    return result_restored

def virtual_try_on(
    input_dict,
    garment_image,
    style_prompt: str = "wearing this clothing, perfectly fitting, realistic cloth texture, natural lighting and folds",
    negative_prompt: str = "naked, deformed, low quality, bad anatomy, distorted folds, unrealistic seams",
    ref_strength: float = 0.8,
    steps: int = 28
):
    """
    Virtual Try-On function:
    - User masks the clothes area on their body.
    - Garment image is fed as visual reference through IP-Adapter.
    """
    if garment_image is None:
        raise ValueError("Harap upload foto pakaian referensi terlebih dahulu.")

    full_prompt = f"realistic fashion photo, {style_prompt}"
    return inpaint_general(
        input_dict=input_dict,
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        reference_image=garment_image,
        ref_strength=ref_strength,
        guidance_scale=8.0,
        num_inference_steps=steps
    )

def replace_background(person_image, new_background, edge_blur: int = 3):
    """
    Instant Background Replacement using rembg:
    - Removes background of person_image
    - Composites person cleanly onto new_background
    """
    from rembg import remove

    if person_image is None:
        raise ValueError("Foto orang belum diunggah.")
    if new_background is None:
        raise ValueError("Foto background baru belum diunggah.")

    if not isinstance(person_image, Image.Image):
        person_image = Image.fromarray(person_image)
    if not isinstance(new_background, Image.Image):
        new_background = Image.fromarray(new_background)

    person_image = person_image.convert("RGB")
    new_background = new_background.convert("RGB")

    # Remove background to get transparent PNG
    print("Extracting person foreground using rembg...")
    cutout = remove(person_image)

    # Resize new background to match person image
    bg_resized = new_background.resize(person_image.size, Image.Resampling.LANCZOS)

    # Smooth the alpha mask edge
    alpha = cutout.split()[3]
    if edge_blur > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(edge_blur))

    # Composite
    bg_resized.paste(cutout, (0, 0), mask=alpha)
    return bg_resized

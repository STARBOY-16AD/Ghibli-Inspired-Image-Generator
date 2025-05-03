import os
import io
import base64
from PIL import Image
import requests
from dotenv import load_dotenv
import torch

# Patch huggingface_hub before importing diffusers
import sys
import importlib.util

# Check if our compatibility module exists
try:
    # Try to import the compatibility module
    import huggingface_compat
    
    # Monkeypatch huggingface_hub
    import huggingface_hub
    huggingface_hub.cached_download = huggingface_compat.cached_download
    print("Applied huggingface_hub compatibility patch")
except ImportError:
    # If the file doesn't exist yet, we'll create it
    print("Creating compatibility module for huggingface_hub")
    compatibility_code = '''
from huggingface_hub import hf_hub_download

# Provide backward compatibility for cached_download
def cached_download(*args, **kwargs):
    """
    Backward compatibility wrapper for cached_download function
    
    This redirects to hf_hub_download which is the modern equivalent
    """
    # Convert legacy parameters if needed
    if 'cache_dir' in kwargs and 'local_dir' not in kwargs:
        kwargs['local_dir'] = kwargs.pop('cache_dir')
    
    if 'pretrained_model_name_or_path' in kwargs and 'repo_id' not in kwargs:
        kwargs['repo_id'] = kwargs.pop('pretrained_model_name_or_path')
    
    if 'filename' in kwargs and 'filename' not in kwargs:
        kwargs['filename'] = kwargs.pop('filename')
    
    # Call the modern function
    return hf_hub_download(*args, **kwargs)
    '''
    
    # Write the compatibility module
    with open('huggingface_compat.py', 'w') as f:
        f.write(compatibility_code)
    
    # Import it
    spec = importlib.util.spec_from_file_location("huggingface_compat", "huggingface_compat.py")
    huggingface_compat = importlib.util.module_from_spec(spec)
    sys.modules["huggingface_compat"] = huggingface_compat
    spec.loader.exec_module(huggingface_compat)
    
    # Monkeypatch huggingface_hub
    import huggingface_hub
    huggingface_hub.cached_download = huggingface_compat.cached_download
    print("Created and applied huggingface_hub compatibility patch")

# Now we can safely import diffusers
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Load environment variables
load_dotenv()

class GhibliArtGenerator:
    def __init__(self):
        """Initialize the Ghibli Art Generator with model loading or API setup"""
        self.api_key = os.getenv("STABILITY_API_KEY") or os.getenv("REPLICATE_API_TOKEN")
        self.use_local_model = os.getenv("USE_LOCAL_MODEL", "False").lower() == "true"
        self.model = None
        
        # If using local model, initialize it
        if self.use_local_model:
            self._setup_local_model()
    
    def _setup_local_model(self):
        """Setup the local Stable Diffusion model"""
        print("Loading local Stable Diffusion model...")
        
        # You could use a fine-tuned model here
        model_id = "runwayml/stable-diffusion-v1-5"  # Base model
        # model_id = "./ghibli_fine_tuned_model"  # If you have a fine-tuned model
        
        # Load model with lower precision for better performance
        try:
            # First try with higher version compatibility
            self.model = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                # Add revision parameter to use a specific version that matches your dependencies
                revision="main"
            )
        except Exception as e:
            print(f"Initial loading failed: {e}")
            # Fallback option with more compatible parameters
            self.model = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                # Removing potentially problematic parameters
                use_safetensors=False
            )
        
        # Use DPM solver for faster inference
        self.model.scheduler = DPMSolverMultistepScheduler.from_config(
            self.model.scheduler.config
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        
        print("Model loaded successfully!")
    
    def generate_with_api(self, prompt):
        """Generate image using external API (Replicate or Stability AI)"""
        if "REPLICATE_API_TOKEN" in os.environ:
            return self._generate_with_replicate(prompt)
        elif "STABILITY_API_KEY" in os.environ:
            return self._generate_with_stability(prompt)
        else:
            raise ValueError("No API keys found for image generation services")
    
    def _generate_with_replicate(self, prompt):
        """Use Replicate.com API to generate images"""
        api_url = "https://api.replicate.com/v1/predictions"
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # You may want to find a model specifically fine-tuned for Ghibli style
        data = {
            "version": "stability-ai/sdxl:8beff3369e81422112d93b89ca01426147de542cd4684c244b673b105188fe5f",  # SDXL model
            "input": {
                "prompt": f"Studio Ghibli style, {prompt}, hand-drawn animation, detailed background, natural lighting, whimsical, fantasy landscape",
                "negative_prompt": "3d, cgi, photorealistic, photography, low quality, bad anatomy",
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
            }
        }
        
        # Create prediction
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        prediction = response.json()
        
        # Poll for results
        get_url = f"https://api.replicate.com/v1/predictions/{prediction['id']}"
        while prediction["status"] not in ["succeeded", "failed"]:
            response = requests.get(get_url, headers=headers)
            response.raise_for_status()
            prediction = response.json()
            if prediction["status"] == "failed":
                raise Exception("Image generation failed")
                
        # Return the image URL
        return prediction["output"][0]
    
    def _generate_with_stability(self, prompt):
        """Use Stability AI API to generate images"""
        api_url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        data = {
            "text_prompts": [
                {
                    "text": f"Studio Ghibli style, {prompt}, hand-drawn animation, detailed background, natural lighting",
                    "weight": 1.0
                },
                {
                    "text": "3d, cgi, photorealistic, photography, low quality, bad anatomy",
                    "weight": -1.0
                }
            ],
            "cfg_scale": 7.5,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 50
        }
        
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        
        # Process and return image data
        result = response.json()
        image_data = result["artifacts"][0]["base64"]
        
        # Save to temp file and return path (or could return base64)
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        img_path = f"static/generated_{os.urandom(8).hex()}.png"
        img.save(img_path)
        
        return img_path
    
    def generate_with_local_model(self, prompt):
        """Generate image using local model"""
        if not self.model:
            raise ValueError("Local model not initialized")
        
        # Enhance prompt with Ghibli-specific details
        enhanced_prompt = f"Studio Ghibli style, {prompt}, hand-drawn animation, detailed background, natural lighting"
        negative_prompt = "3d, cgi, photorealistic, photography, low quality, bad anatomy"
        
        # Generate image
        with torch.no_grad():
            image = self.model(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
            ).images[0]
        
        # Save image
        img_path = f"static/generated/{os.urandom(8).hex()}.png"
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        image.save(img_path)
        
        return img_path
    
    def generate(self, prompt):
        """Generate art based on prompt"""
        try:
            if self.use_local_model:
                return self.generate_with_local_model(prompt)
            else:
                return self.generate_with_api(prompt)
        except Exception as e:
            print(f"Error generating image: {e}")
            raise

# For testing
if __name__ == "__main__":
    generator = GhibliArtGenerator()
    result = generator.generate("A peaceful forest with small spirits hiding among trees and flowers")
    print(f"Generated image: {result}")
#!/usr/bin/env python3
"""
Service API spécialisé pour la génération d'images avec SDXL
Déployable sur serveur GPU distant
"""

import os
import base64
import io
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Creatures AI Generation Service", 
    version="1.0.0",
    description="Service spécialisé pour la génération d'images avec SDXL"
)

# CORS pour appels depuis le backend local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AIGenerationService:
    def __init__(self):
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Charge SDXL avec optimisations GPU"""
        if self.pipeline is None:
            logger.info("Loading SDXL img2img pipeline...")
            
            self.pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            ).to(self.device)
            
            # Optimisations GPU
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
                self.pipeline.enable_vae_slicing()
                self.pipeline.enable_attention_slicing()
            
            logger.info("SDXL loaded successfully")
    
    def generate_image(
        self,
        input_image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        strength: float = 0.7,
        width: int = 1024,
        height: int = 1024
    ) -> str:
        """Génère une image avec SDXL img2img"""
        try:
            self.load_model()
            
            # Redimensionner l'image d'entrée
            input_image = input_image.resize((width, height))
            
            logger.info(f"Generating with prompt: {prompt[:100]}...")
            
            # Génération SDXL
            generated_images = self.pipeline(
                prompt=prompt,
                image=input_image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                width=width,
                height=height
            ).images
            
            generated_image = generated_images[0]
            
            # Convertir en base64
            buffer = io.BytesIO()
            generated_image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Nettoyage mémoire
            del generated_images, generated_image
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Instance du service
ai_service = AIGenerationService()

@app.get("/")
async def root():
    return {
        "service": "Creatures AI Generation Service",
        "status": "running",
        "device": ai_service.device
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "device": ai_service.device}

@app.post("/img2img")
async def img2img_generation(
    image: UploadFile = File(..., description="Image d'entrée"),
    prompt: str = Form(..., description="Prompt de génération"),
    negative_prompt: str = Form(default="", description="Prompt négatif"),
    num_inference_steps: int = Form(default=20, description="Nombre d'étapes"),
    guidance_scale: float = Form(default=7.5, description="Force du guidage"),
    strength: float = Form(default=0.7, description="Force de transformation"),
    width: int = Form(default=1024, description="Largeur de sortie"),
    height: int = Form(default=1024, description="Hauteur de sortie")
):
    """
    Endpoint principal pour la génération img2img avec SDXL
    """
    try:
        # Lire l'image uploadée
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        logger.info(f"Processing image {pil_image.size} with prompt: {prompt[:50]}...")
        
        # Génération
        image_base64 = ai_service.generate_image(
            input_image=pil_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            width=width,
            height=height
        )
        
        return JSONResponse({
            "success": True,
            "image_base64": image_base64,
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "strength": strength,
                "resolution": f"{width}x{height}"
            }
        })
        
    except Exception as e:
        logger.error(f"Error in img2img endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Port différent pour éviter les conflits
    uvicorn.run(app, host="0.0.0.0", port=8080)
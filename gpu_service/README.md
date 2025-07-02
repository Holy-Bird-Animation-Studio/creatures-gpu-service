# Service API de Génération IA

Service spécialisé pour la génération d'images avec SDXL, déployable sur serveur GPU.

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
python main.py
```

Le service sera accessible sur `http://0.0.0.0:8080`

## API Endpoints

### POST /img2img
Génération d'image à partir d'une image d'entrée

**Paramètres:**
- `image`: Image d'entrée (form-data)
- `prompt`: Prompt de génération
- `negative_prompt`: Prompt négatif (optionnel)
- `num_inference_steps`: Nombre d'étapes (défaut: 20)
- `guidance_scale`: Force du guidage (défaut: 7.5)
- `strength`: Force de transformation (défaut: 0.7)
- `width`: Largeur de sortie (défaut: 1024)
- `height`: Hauteur de sortie (défaut: 1024)

**Réponse:**
```json
{
  "success": true,
  "image_base64": "base64_encoded_image",
  "parameters": {...}
}
```

### GET /health
Vérification de l'état du service

## Déploiement

1. Serveur avec GPU (RTX 3080+ recommandé)
2. CUDA 11.8+
3. 16GB+ VRAM pour SDXL
4. Ouverture du port 8080

## Sécurité

⚠️ En production:
- Ajouter authentification API
- Restreindre CORS
- Limiter la taille des uploads
- Rate limiting
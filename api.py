import os
import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

app = FastAPI()

# Create the gen_art directory if it doesn't exist
gen_art_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/gen_art")
os.makedirs(gen_art_dir, exist_ok=True)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate_image/")
def generate_image(request: Request, prompt_request: PromptRequest):
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        image_path = os.path.join(gen_art_dir, f"generated_image_{timestamp}.png")
        image = pipe(prompt_request.prompt).images[0]
        image.save(image_path)
        image_url = f"/assets/gen_art/generated_image_{timestamp}.png"
        return {"image_url": image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assets/{file_path:path}")
def read_file(file_path: str):
    return FileResponse(os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", file_path))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

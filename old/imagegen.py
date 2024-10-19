from diffusers import AutoPipelineForText2Image
import time
import torch
from PIL import Image

# 모델 로드
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

# 프롬프트 설정
# prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
prompt = '''Design a front side of credit card. Show me just one credit card. This card is for 40s housewives with bright colors and elegant atmospheres.
            "Lotte Card" name should be placed on the card. Make the card design sharp and clear. Don't misspell "Lotte". Noble woman portrait recommended'''

# 이미지 생성
image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0).images[0]

# 이미지 저장
time_stamp = int(time.time())
image.save(f"output/generated_image_{time_stamp}.png")
from diffusers import AutoPipelineForText2Image
import time
import torch
from PIL import Image

# 모델 로드
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

# 프롬프트 설정
# prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
prompt = "Design a credit card of Lotte Card for 40s housewives with a brightly colored and gorgeous image."

# 이미지 생성
image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

# 이미지 저장
time_stamp = int(time.time())
image.save(f"output/generated_image_{time_stamp}.png")
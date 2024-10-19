from PIL import Image
import pytesseract
import os

# 이미지 파일 경로
img = input('input image name: ' )
img_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), f"data/{img}")

# 이미지 열기
image = Image.open(img_path)

# 한글 인식
text = pytesseract.image_to_string(image, lang='kor')
print(text)

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"output/{img.split('.')[0]}.txt"), 'w') as f:
    f.write(text)

# tesseract 00110011001.jpg output_name -l kor
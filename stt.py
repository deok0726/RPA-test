import os
import numpy as np
import speech_recognition as sr

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

def mp3_to_wav(file):
    # files                                                                         
    src = file
    dst = 'test.wav'

    if os.path.isfile(src):
        print('file exists')

    # convert wav to mp3                                                            
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")

def speech_to_text(audio):
    # 음성 인식 객체 생성
    recognizer = sr.Recognizer()
    # 음성 파일 경로
    audio_file_path = audio

    # 음성 파일 불러오기
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
    
    # Google Web Speech API를 사용하여 음성을 텍스트로 변환
    time_stamp = time.time()
    try:
        text = recognizer.recognize_google(audio_data, language="ko-KR")
        print("인식된 텍스트:", text)
        with open(f"output/stt_{time_stamp}.txt", "w") as file:
            file.write(text)
    except sr.UnknownValueError:
        print("음성을 인식하지 못했습니다.")
    except sr.RequestError as e:
        print(f"Google Web Speech API 요청 에러: {e}")

def real_time_speech_to_text():
    init_rec = sr.Recognizer()

    print("Let's speak!!")
    with sr.Microphone(device_index=0) as source:
        audio_data = init_rec.record(source, duration=10)
        print("Recognizing voice.............")
        text = init_rec.recognize_google(audio_data, language="ko-KR")
        print(f"Recognized text: {text}")
        
        # Save the recognized text to a file
        time_stamp = time.time()
        with open("output/rtstt_{time_stamp}.txt", "w") as file:
            file.write(text)


if __name__ == "__main__":
    # real_time_speech_to_text()
    # mp3_to_wav('test.mp3')
    # speech_to_text('test.wav')
    # diarization('test.wav')
    # sentiment_classifier("output2.txt")
import os
import time
import shutil
from pydub import AudioSegment
import speech_recognition as sr
from pyAudioAnalysis import audioSegmentation as aS

def stt_diarization(file):
    # 오디오 파일을 로드합니다
    audio_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), f"data/{file}")

    # 입력 오디오 파일 형식
    # audio = AudioSegment.from_mp3(audio_file)
    audio = AudioSegment.from_file(audio_file, format='m4a')

    # 오디오 파일을 wav 형식으로 변환합니다 (SpeechRecognition 라이브러리는 wav 형식을 사용합니다)
    audio.export("data/converted_audio.wav", format="wav")

    # 화자 분류를 수행합니다
    flags, purity_cluster_m, purity_speaker_m = aS.speaker_diarization("data/converted_audio.wav", n_speakers=2)

    # 화자 분류 결과를 기반으로 오디오를 분할합니다
    segments = []
    start = 0
    for i, flag in enumerate(flags):
        if i == 0 or flag != flags[i - 1]:
            if i != 0:
                segments.append((start, i))
            start = i
    segments.append((start, len(flags)))

    # SpeechRecognition을 사용하여 음성을 텍스트로 변환합니다
    recognizer = sr.Recognizer()
    time_stamp = int(time.time())

    os.makedirs(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"segment_{time_stamp}"),  exist_ok=True)

    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"output/diarized_{time_stamp}.txt"), 'w', encoding='utf-8') as f:
        for i, (start, end) in enumerate(segments):
            segment_audio = audio[start*1000:end*1000]  # pyAudioAnalysis는 초 단위, pydub은 밀리초 단위를 사용합니다
            segment_audio.export(f"segment_{time_stamp}/segment_{i}.wav", format="wav")
            
            with sr.AudioFile(f"segment_{time_stamp}/segment_{i}.wav") as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data, language="ko-KR")
                    print(f"화자 {flags[start]}: {text}")
                    f.write(f"화자 {flags[start]}: {text}\n")
                except sr.UnknownValueError:
                    print(f"화자 {flags[start]}: 음성을 인식할 수 없습니다.")
                except sr.RequestError as e:
                    print(f"화자 {flags[start]}: 음성 인식 서비스에 접근할 수 없습니다; {e}")

    shutil.rmtree(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"segment_{time_stamp}"))

if __name__ == "__main__":
    # stt_diarization("test.mp3")
    stt_diarization("shopping/쇼핑_3863.m4a")
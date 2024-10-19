import os
import sys
import shutil
from pydub import AudioSegment
from speechlib import Transcriptor

def speechlib_diarization(path):
    # audio_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), f"data/{path}")
    split_string = path.split('/')
    last_part = split_string[-1]
    split_string = last_part.split('.')
    
    file_name = split_string[0]
    file_format = split_string[-1]

    audio = AudioSegment.from_file(path, format=f'{file_format}')
    audio.export(f"data/converted/{file_name}.wav", format="wav")

    file = f"data/converted/{file_name}.wav"  # your audio file
    voices_folder = "" # voices folder containing voice samples for recognition
    language = "ko"          # language code
    log_folder = "output"      # log folder for storing transcripts
    modelSize = "large-v3"     # size of model to be used [tiny, small, medium, large-v1, large-v2, large-v3]
    quantization = False   # setting this 'True' may speed up the process but lower the accuracy
    ACCESS_TOKEN = "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" # get permission to access pyannote/speaker-diarization@2.1 on huggingface

    # quantization only works on faster-whisper
    transcriptor = Transcriptor(file, log_folder, language, modelSize, ACCESS_TOKEN, voices_folder, quantization)

    # res = transcriptor.whisper()
    res = transcriptor.faster_whisper()

    # res --> [["start", "end", "text", "speaker"], ["start", "end", "text", "speaker"]...]
    
    output = "/svc/project/genaipilot/output"
    # rename output and remove converted
    for file in os.listdir(output):
        if file_name in file:
            os.rename(f"{output}/{file}", f"{output}/{file_name}.txt")
            os.system(f"rm data/converted/{file_name}.wav")
            return

if __name__ == "__main__":
    # python speechlibdiar.py finance/balance
    
    input_file = sys.argv[1]
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), f"data/{input_file}")
    if os.path.isdir(path):
        print(f"{path} is a directory.")
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for f in files:
            print(path + '/' + f)
            file = path + '/' + f
            speechlib_diarization(file)
    else:
        print(f"{path} is a file.")
        speechlib_diarization(path)
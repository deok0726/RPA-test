import re, os, sys, time
import torch
import webvtt
import whisper
from datetime import timedelta
from pydub import AudioSegment
from pyannote.audio import Pipeline


def millisec(timeStr):
  spl = timeStr.split(":")
  s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  return s

def whisper_diarization(file):
    audio_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), f"data/{file}")

    audio = AudioSegment.from_file(audio_file, format='m4a')
    audio.export("data/converted_audio.wav", format="wav")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
    # pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization')

    # DEMO_FILE = {'uri': 'blabal', 'audio': f'{file}'}
    # dz = pipeline(DEMO_FILE)

    pipeline.to(torch.device("cuda"))
    dz = pipeline("data/converted_audio.wav", num_speakers=2)

    time_stamp = int(time.time())
    with open(f"output/diarization_{time_stamp}.txt", "w") as text_file:
        text_file.write(str(dz))

    # print(*list(dz.itertracks(yield_label = True))[:10], sep="\n")
    for turn, _, speaker in dz.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    
    dz = open(f'output/diarization_{time_stamp}.txt').read().splitlines()
    dzList = []
    spacermilli = 2000
    spacer = AudioSegment.silent(duration=spacermilli)

    for l in dz:
        start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
        start = millisec(start) - spacermilli
        end = millisec(end)  - spacermilli
        guest = not re.findall('SPEAKER_00', string=l)
        dzList.append([start, end, guest])

    print(*dzList[:10], sep='\n')

    sounds = spacer
    segments = []

    dz = open(f'output/diarization_{time_stamp}.txt').read().splitlines()
    for l in dz:
        start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
        start = int(millisec(start)) #milliseconds
        end = int(millisec(end))  #milliseconds
        
        segments.append(len(sounds))
        sounds = sounds.append(audio[start:end], crossfade=0)
        sounds = sounds.append(spacer, crossfade=0)

    sounds.export(f"output/dz_{time_stamp}.wav", format="wav") #Exports to a wav file in the current path.
    os.system(f"rm output/diarization_{time_stamp}.txt")
    print(segments[:8])

    # model = whisper.load_model("base", language='ko')
    # result = model.transcribe(f"output/dz_{time_stamp}.wav")
    os.system(f"whisper output/dz_{time_stamp}.wav --language Korean")
    os.system(f"mv dz_{time_stamp}.* output")

    captions = [[(int)(millisec(caption.start)), (int)(millisec(caption.end)),  caption.text] for caption in webvtt.read(f'output/dz_{time_stamp}.vtt')]
    print(*captions[:8], sep='\n')

    preS = '\n\n  \n    \n    \n    \n    LOCA\n    \n  \n  \n    Customer Service STT\n  \n    \n'
    postS = '\t\n'
    html = list(preS)

    for i in range(len(segments)):
        idx = 0
        for idx in range(len(captions)):
            if captions[idx][0] >= (segments[i] - spacermilli):
                break;
        
        while (idx < (len(captions))) and ((i == len(segments) - 1) or (captions[idx][1] < segments[i+1])):
            c = captions[idx]  
            
            start = dzList[i][0] + (c[0] -segments[i])

            if start < 0: 
                start = 0
            idx += 1

            start = start / 1000.0
            startStr = '{0:02d}:{1:02d}:{2:02.2f}'.format((int)(start // 3600), 
                                                    (int)(start % 3600 // 60), 
                                                    start % 60)
            
            html.append('\t\t\t\n')
            html.append(f'\t\t\t\tlink |\n')
            html.append(f'\t\t\t\t{startStr}\n')
            html.append(f'\t\t\t\t{"[Guest]" if dzList[i][2] else "[Counseler]"} {c[2]}\n')
            html.append('\t\t\t\n\n')

    html.append(postS)
    s = "".join(html)

    with open(f"output/CS_{time_stamp}.html", "w") as text_file:
        text_file.write(s)
    print(s)

    os.system(f"rm output/dz_{time_stamp}.*")

if __name__ == "__main__":
    whisper_diarization("shopping/쇼핑_1995.m4a")
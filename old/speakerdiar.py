import os
import time
import torch
import torchaudio
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

def speaker_diarization(file):

    audio_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), f"data/{file}")

    # audio = AudioSegment.from_mp3(audio_file)
    audio = AudioSegment.from_file(audio_file, format='m4a')
    audio.export("data/converted_audio.wav", format="wav")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", 
                                        use_auth_token="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")

    pipeline.to(torch.device("cuda"))

    waveform, sample_rate = torchaudio.load("data/converted_audio.wav")
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    with ProgressHook() as hook:
        diarization = pipeline("data/converted_audio.wav", hook=hook)

    diarization = pipeline("data/converted_audio.wav", num_speakers=2)

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

    time_stamp = int(time.time())
    # dump the diarization output to disk using RTTM format
    with open(f"output/audio_{time_stamp}.rttm", "w") as rttm:
        diarization.write_rttm(rttm)

if __name__ == "__main__":
    speaker_diarization("call/쇼핑_27.m4a")
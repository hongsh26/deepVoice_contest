import os
from pyannote.audio import Pipeline
from pydub import AudioSegment
import numpy as np


def diarize(file, pipeline):
    # 오디오 파일 경로 설정
    audio_file = "../data/test_nonBack/" + file

    # 스피커 다이어리제이션 수행
    diarization = pipeline(audio_file)

    # 오디오 파일 로드
    audio = AudioSegment.from_wav(audio_file)

    # 화자별로 음성을 분리하여 저장
    output_dir = "../data/test_output_speakers"
    os.makedirs(output_dir, exist_ok=True)

    # temp 폴더 생성
    temp_dir = "../data/temp_segments"
    os.makedirs(temp_dir, exist_ok=True)

    speaker_audio_segments = {}
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_audio_segments:
            speaker_audio_segments[speaker] = []
        segment_audio = audio[segment.start * 1000: segment.end * 1000]
        segment_path = os.path.join(temp_dir, f"{speaker}_{segment.start:.1f}_{segment.end:.1f}.wav")
        segment_audio.export(segment_path, format="wav")
        speaker_audio_segments[speaker].append(segment_path)

    # Demucs를 사용하여 각 화자의 목소리를 분리하고 증폭
    def separate_and_amplify(file_path, target_factor=2.0):
        os.system(f"demucs -n mdx_extra_q {file_path} -o {temp_dir}")
        separated_path = os.path.join(temp_dir, os.path.basename(file_path))
        vocals_audio = AudioSegment.from_wav(separated_path)
        vocals_audio = vocals_audio.apply_gain(10 * np.log10(target_factor))
        return vocals_audio

    count = 0
    for speaker, segments in speaker_audio_segments.items():
        count += 1
        combined_audio = AudioSegment.empty()
        for segment_path in segments:
            processed_audio = separate_and_amplify(segment_path)
            combined_audio += processed_audio

        output_path = os.path.join(output_dir, f"{file.replace('.wav', '')}_{count}.wav")
        combined_audio.export(output_path, format="wav")
        print(f"Saved processed audio for speaker {speaker} to {output_path}")


files = os.listdir("../data/test_nonBack")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token='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')
for file in files:
    diarize(file, pipeline)
import shutil

temp_dir = "../data/temp_segments"
shutil.rmtree(temp_dir)

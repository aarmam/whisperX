from io import IOBase
from pathlib import Path
from typing import Mapping, Text
from typing import Optional
from typing import Union

import torch

from whisperx.diarize import Segment as SegmentX
from whisperx.vads.vad import Vad

AudioFile = Union[Text, Path, IOBase, Mapping]


class Silero(Vad):
    _model_cache = None
    _utils_cache = None

    # check again default values
    def __init__(self, **kwargs):
        print(">>Performing voice activity detection using Silero...")
        super().__init__(kwargs['vad_onset'])

        self.vad_onset = kwargs['vad_onset']
        self.chunk_size = kwargs['chunk_size']

        # Use cached model if available
        if Silero._model_cache is None:
            Silero._model_cache, vad_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True
            )
            Silero._utils_cache = vad_utils

        self.vad_pipeline = Silero._model_cache
        (self.get_speech_timestamps, _, self.read_audio, _, _) = Silero._utils_cache

    def __call__(self, audio: AudioFile, **kwargs):
        """use silero to get segments of speech"""
        # Only accept 16000 Hz for now.
        # Note: Silero models support both 8000 and 16000 Hz. Although other values are not directly supported,
        # multiples of 16000 (e.g. 32000 or 48000) are cast to 16000 inside of the JIT model!
        sample_rate = audio["sample_rate"]
        if sample_rate != 16000:
            raise ValueError("Only 16000Hz sample rate is allowed")

        # Optimize parameters for faster processing
        timestamps = self.get_speech_timestamps(
            audio["waveform"],
            model=self.vad_pipeline,
            sampling_rate=sample_rate,
            max_speech_duration_s=self.chunk_size,
            threshold=self.vad_onset,
            min_silence_duration_ms=100,  # Reduce unnecessary splits
            min_speech_duration_ms=250,   # Filter out very short segments
            return_seconds=False,          # Slightly faster
            visualize_probs=False,         # Disable visualization
            progress_tracking_callback=None
        )
        return [SegmentX(i['start'] / sample_rate, i['end'] / sample_rate, "UNKNOWN") for i in timestamps]

    @staticmethod
    def preprocess_audio(audio):
        return audio

    @staticmethod
    def merge_chunks(segments_list,
                     chunk_size,
                     onset: float = 0.5,
                     offset: Optional[float] = None,
                     ):
        assert chunk_size > 0
        if len(segments_list) == 0:
            print("No active speech found in audio")
            return []
        assert segments_list, "segments_list is empty."
        return Vad.merge_chunks(segments_list, chunk_size, onset, offset)

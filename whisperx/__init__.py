import importlib


def _lazy_import(name):
    module = importlib.import_module(f"whisperx.{name}")
    return module


def load_align_model(*args, **kwargs):
    alignment = _lazy_import("alignment")
    return alignment.load_align_model(*args, **kwargs)


def align(*args, **kwargs):
    alignment = _lazy_import("alignment")
    return alignment.align(*args, **kwargs)


def load_model(*args, **kwargs):
    asr = _lazy_import("asr")
    return asr.load_model(*args, **kwargs)


def load_audio(*args, **kwargs):
    audio = _lazy_import("audio")
    return audio.load_audio(*args, **kwargs)


def assign_word_speakers(*args, **kwargs):
    diarize = _lazy_import("diarize")
    return diarize.assign_word_speakers(*args, **kwargs)


def process_long_audio_streaming(*args, **kwargs):
    streaming = _lazy_import("streaming")
    return streaming.process_long_audio_streaming(*args, **kwargs)


def setup_mixed_precision(*args, **kwargs):
    acceleration = _lazy_import("acceleration")
    return acceleration.setup_mixed_precision(*args, **kwargs)


def optimize_torch_settings(*args, **kwargs):
    acceleration = _lazy_import("acceleration")
    return acceleration.optimize_torch_settings(*args, **kwargs)


def get_optimal_batch_size(*args, **kwargs):
    acceleration = _lazy_import("acceleration")
    return acceleration.get_optimal_batch_size(*args, **kwargs)

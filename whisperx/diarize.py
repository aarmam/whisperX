import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
from typing import Optional, Union
import torch

from whisperx.audio import load_audio, SAMPLE_RATE
from whisperx.types import TranscriptionResult, AlignedTranscriptionResult


class DiarizationPipeline:
    def __init__(
        self,
        model_name=None,
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        model_config = model_name or "pyannote/speaker-diarization-3.1"
        self.model = Pipeline.from_pretrained(model_config, use_auth_token=use_auth_token).to(device)

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
    ) -> Union[tuple[pd.DataFrame, Optional[dict[str, list[float]]]], pd.DataFrame]:
        """
        Perform speaker diarization on audio.

        Args:
            audio: Path to audio file or audio array
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            return_embeddings: Whether to return speaker embeddings

        Returns:
            If return_embeddings is True:
                Tuple of (diarization dataframe, speaker embeddings dictionary)
            Otherwise:
                Just the diarization dataframe
        """
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }

        if return_embeddings:
            diarization, embeddings = self.model(
                audio_data,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=True,
            )
        else:
            diarization = self.model(
                audio_data,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            embeddings = None

        diarize_df = pd.DataFrame(diarization.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)

        if return_embeddings and embeddings is not None:
            speaker_embeddings = {speaker: embeddings[s].tolist() for s, speaker in enumerate(diarization.labels())}
            return diarize_df, speaker_embeddings
        
        # For backwards compatibility
        if return_embeddings:
            return diarize_df, None
        else:
            return diarize_df


def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: Union[AlignedTranscriptionResult, TranscriptionResult],
    speaker_embeddings: Optional[dict[str, list[float]]] = None,
    fill_nearest: bool = False,
) -> Union[AlignedTranscriptionResult, TranscriptionResult]:
    """
    Assign speakers to words and segments in the transcript.

    Args:
        diarize_df: Diarization dataframe from DiarizationPipeline
        transcript_result: Transcription result to augment with speaker labels
        speaker_embeddings: Optional dictionary mapping speaker IDs to embedding vectors
        fill_nearest: If True, assign speakers even when there's no direct time overlap

    Returns:
        Updated transcript_result with speaker assignments and optionally embeddings
    """
    transcript_segments = transcript_result["segments"]

    if diarize_df.empty:
        if speaker_embeddings is not None:
            transcript_result["speaker_embeddings"] = speaker_embeddings
        return transcript_result

    ordered_df = diarize_df.sort_values("start", kind="stable").reset_index(drop=True)
    starts = ordered_df['start'].to_numpy()
    ends = ordered_df['end'].to_numpy()
    speakers = ordered_df['speaker'].to_numpy()

    def dominant_speaker(start_time: float, end_time: float) -> Optional[str]:
        overlaps = np.minimum(ends, end_time) - np.maximum(starts, start_time)
        if overlaps.size == 0:
            return None

        positive_mask = overlaps > 0
        if np.any(positive_mask):
            totals: dict[str, float] = {}
            for speaker, overlap in zip(speakers[positive_mask], overlaps[positive_mask]):
                totals[speaker] = totals.get(speaker, 0.0) + float(overlap)
            return max(totals.items(), key=lambda item: item[1])[0]

        if fill_nearest:
            nearest_idx = int(np.argmax(overlaps))
            return speakers[nearest_idx]

        return None

    for seg in transcript_segments:
        speaker = dominant_speaker(seg['start'], seg['end'])
        if speaker is not None:
            seg["speaker"] = speaker

        if 'words' in seg:
            for word in seg['words']:
                if 'start' in word and 'end' in word:
                    word_speaker = dominant_speaker(word['start'], word['end'])
                    if word_speaker is not None:
                        word["speaker"] = word_speaker

    # Add speaker embeddings to the result if provided
    if speaker_embeddings is not None:
        transcript_result["speaker_embeddings"] = speaker_embeddings

    return transcript_result


class Segment:
    def __init__(self, start:int, end:int, speaker:Optional[str]=None):
        self.start = start
        self.end = end
        self.speaker = speaker

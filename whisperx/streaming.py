"""
Streaming audio processing utilities for WhisperX
Enables processing of long audio files with minimal memory usage
"""

import gc
import numpy as np
import torch
from typing import Iterator, Optional, Union, Generator
from whisperx.audio import load_audio, SAMPLE_RATE
from whisperx.types import TranscriptionResult, SingleSegment


class StreamingAudioProcessor:
    """
    Process audio in streaming chunks to reduce memory usage for long files
    """

    def __init__(self, chunk_duration: float = 300.0, overlap_duration: float = 30.0):
        """
        Initialize streaming processor

        Args:
            chunk_duration: Duration of each chunk in seconds (default 5 minutes)
            overlap_duration: Overlap between chunks in seconds (default 30 seconds)
        """
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = SAMPLE_RATE

    def stream_audio_chunks(self, audio_path: str) -> Generator[tuple[np.ndarray, float, float], None, None]:
        """
        Stream audio file in overlapping chunks

        Args:
            audio_path: Path to audio file

        Yields:
            Tuple of (audio_chunk, start_time, end_time)
        """
        # Load audio file info without loading entire file
        audio = load_audio(audio_path)
        total_duration = len(audio) / self.sample_rate

        chunk_samples = int(self.chunk_duration * self.sample_rate)
        overlap_samples = int(self.overlap_duration * self.sample_rate)

        start_sample = 0

        while start_sample < len(audio):
            end_sample = min(start_sample + chunk_samples, len(audio))

            # Extract chunk
            chunk = audio[start_sample:end_sample]
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate

            yield chunk, start_time, end_time

            # Move to next chunk with overlap
            start_sample += chunk_samples - overlap_samples

            # Avoid tiny final chunks
            if len(audio) - start_sample < overlap_samples:
                break

        # Clean up
        del audio
        gc.collect()

    def merge_overlapping_results(self, results: list[tuple[TranscriptionResult, float, float]]) -> TranscriptionResult:
        """
        Merge results from overlapping chunks, removing duplicates

        Args:
            results: List of (result, start_time, end_time) tuples

        Returns:
            Merged transcription result
        """
        if not results:
            return {"segments": [], "language": "en"}

        merged_segments = []
        language = results[0][0]["language"]

        for result, chunk_start_time, chunk_end_time in results:
            for segment in result["segments"]:
                # Adjust segment times to global timeline
                adjusted_segment = segment.copy()
                adjusted_segment["start"] += chunk_start_time
                adjusted_segment["end"] += chunk_start_time

                # Skip segments that are likely duplicates from overlap
                is_duplicate = False
                for existing_segment in merged_segments:
                    # Check for significant overlap
                    overlap_start = max(existing_segment["start"], adjusted_segment["start"])
                    overlap_end = min(existing_segment["end"], adjusted_segment["end"])
                    overlap_duration = max(0, overlap_end - overlap_start)

                    segment_duration = adjusted_segment["end"] - adjusted_segment["start"]
                    existing_duration = existing_segment["end"] - existing_segment["start"]

                    # If more than 70% overlap, consider it a duplicate
                    if (overlap_duration / min(segment_duration, existing_duration)) > 0.7:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    merged_segments.append(adjusted_segment)

        # Sort segments by start time
        merged_segments.sort(key=lambda x: x["start"])

        return {"segments": merged_segments, "language": language}


def process_long_audio_streaming(
    audio_path: str,
    model,
    chunk_duration: float = 300.0,
    overlap_duration: float = 30.0,
    **transcribe_kwargs
) -> TranscriptionResult:
    """
    Process long audio file using streaming approach

    Args:
        audio_path: Path to audio file
        model: WhisperX model pipeline
        chunk_duration: Duration of each chunk in seconds
        overlap_duration: Overlap between chunks in seconds
        **transcribe_kwargs: Additional arguments for transcription

    Returns:
        Complete transcription result
    """
    processor = StreamingAudioProcessor(chunk_duration, overlap_duration)
    results = []

    for chunk_idx, (audio_chunk, start_time, end_time) in enumerate(processor.stream_audio_chunks(audio_path)):
        print(f"Processing chunk {chunk_idx + 1}: {start_time:.1f}s - {end_time:.1f}s")

        # Transcribe chunk
        result = model.transcribe(audio_chunk, **transcribe_kwargs)
        results.append((result, start_time, end_time))

        # Force garbage collection after each chunk
        del audio_chunk
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Merge results
    return processor.merge_overlapping_results(results)
# Canonical Test Assets

Use these assets for orchestrated multimodal validation in `tests/full_matrix_benchmark.py`.

## Primary assets

1. PowerPoint (PPTX)
   - `Overview of International Business.pptx`
2. Video (MP4)
   - `Real-Time, Low Latency and High Temporal Resolution Spectrograms - Alexandre R.J. Francois - ADC.mp4`

## Purpose

- PPTX validates slide parsing, text chunking, image handling, retrieval, and graph reasoning.
- Video validates frame extraction + audio/transcription path in API mode and multimodal response quality.

## Notes

- If the PPTX asset is missing, benchmark tooling falls back to `README.md` for continuity.
- If the video asset is missing, video runs are skipped and reported as skipped.

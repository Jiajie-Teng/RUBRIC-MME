# RUBRIC-MME Tools

This folder contains development-time helpers for RUBRIC-MME. The benchmark itself still lives in `lmms_eval/tasks/omnibench`, while these scripts help us validate the Phase 1 multi-round generation chain without depending on a local GPU.

## Data Layout

The current local development layout assumes that both JSON files and all media files live under:

- `D:/lmms-eval/omnibench_dataset`

In particular:

- `D:/lmms-eval/omnibench_dataset/image_final_with_mimt_category.json`
- `D:/lmms-eval/omnibench_dataset/video_final_with_vqa_category.json`

## `run_gemini_phase1.py`

Runs RUBRIC-MME Phase 1 generation with the Gemini API and writes structured outputs to JSONL files.

### Example

```bash
python tools/RUBRIC-MME/run_gemini_phase1.py \
  --tasks omnibench_image_multi_text \
  --model gemini-2.5-flash \
  --data-root ./omnibench_dataset \
  --limit 1 \
  --output-dir ./logs/rubric_mme_gemini_phase1
```

To run all four RUBRIC-MME variants:

```bash
python tools/RUBRIC-MME/run_gemini_phase1.py \
  --tasks omnibench \
  --model gemini-2.5-flash \
  --data-root ./omnibench_dataset \
  --output-dir ./logs/rubric_mme_gemini_phase1 \
  --resume
```

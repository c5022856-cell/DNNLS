# Multimodal Alignment for Visual Story Continuation

Purpose

- Align predicted captions with target images using shared CLIP-based embeddings and a prefix-conditioned GPT-2 decoder.

Method overview

- Dataset: StoryReasoning, with per-frame captions parsed from `<gdi imageX>` segments.
- Encoder: CLIP image + CLIP text embeddings fused and temporally aggregated via a Transformer encoder.
- Decoder: GPT-2 with prefix conditioning from the shared context vector.
- Losses: text loss, image embedding regression, and cosine alignment between predicted image and text embeddings.

Project summary

- Task: K+1 visual story continuation (K=4). Given 4 context frames (images + captions), predict the next caption and align it to the target image.
- Data integrity: frame_count matches number of images for all samples; missing per-frame caption segments are rare.
- Caption style: long narrative sentences; average ~80 words per frame, which motivates prompt compression.

Dataset statistics (train split)

| Metric | Value |
|---|---:|
| Train size | 3,552 |
| Frame count min / max | 5 / 22 |
| Mean frame count | 12.4434 |
| Median frame count | 13 |
| Avg words per frame (mean) | 80.1926 |
| Avg words per frame (median) | 78.2426 |
| Avg missing segments | 0.000282 |
| frame_count == num_images | 100% |
| % with >= 5 frames (K+1) | 100% |

Split strategy (K=4, length-bin stratified by frame_count)

| Split | Size |
|---|---:|
| Train | 2,841 |
| Val | 355 |
| Test | 356 |

Modeling results

- Text-only baseline (GPT-2, 1 epoch): train loss 2.2318, val loss 2.0305.
- Project4 (frozen GPT-2, 2 epochs): multi-loss training reduces total loss while learning image-text alignment.
- Project4 v2: alignment loss uses frozen CLIP text embedding of the GT caption to avoid collapse.

Training losses (Project4, 2 epochs)

| Epoch | Split | txt_loss | img_loss | align_loss | total_loss |
|---|---|---:|---:|---:|---:|
| 1 | Train | 2.8431 | 0.00345 | 0.01863 | 2.8542 |
| 1 | Val | 2.5619 | 0.00298 | 0.00083 | 2.5638 |
| 2 | Train | 2.7522 | 0.00253 | 0.00050 | 2.7537 |
| 2 | Val | 2.5441 | 0.00211 | 0.00052 | 2.5455 |

Evaluation (Project4, no generation)

| Metric | Val | Test |
|---|---:|---:|
| cos(pred_img, tgt_img) | 0.45924 | 0.45791 |
| cos(pred_img, txt_emb) | 0.99948 | 0.99948 |
| cos(tgt_img, GT_CLIPtxt) | 0.24948 | 0.25230 |
| txt_loss | 2.54359 | 2.55898 |

Generated caption CLIPScore (Project4, 30 batches)

| Split | CLIPScore (tgt_img vs generated caption) |
|---|---:|
| Val | 0.24685 |
| Test | 0.24265 |

Training losses (Project4 v2, 2 epochs)

| Epoch | Split | txt_loss | img_loss | align_loss | total_loss |
|---|---|---:|---:|---:|---:|
| 1 | Train | 2.8375 | 0.00298 | 0.17768 | 2.9278 |
| 1 | Val | 2.5519 | 0.00294 | 0.15741 | 2.6321 |
| 2 | Train | 2.7401 | 0.00296 | 0.15068 | 2.8169 |
| 2 | Val | 2.5326 | 0.00298 | 0.15578 | 2.6120 |

Interpretation

- The shared encoder learns to predict CLIP image embeddings for the next frame with moderate cosine similarity (~0.458).
- CLIP alignment between target images and GT captions is ~0.25, which provides a realistic reference point for generated caption CLIPScores (~0.243-0.247).
- v2 alignment increases the alignment loss magnitude due to a stronger, frozen CLIP text target, which is expected and more stable than the near-1.0 model-dependent alignment.

Repository layout

- `src/dataloader.py`: caption parsing, stratified split, dataset and collation.
- `src/encoders_image.py`: CLIP image embedding helper.
- `src/encoders_text.py`: GPT-2 tokenization and CLIP text embeddings.
- `src/generator.py`: shared encoder, prefix GPT-2, and alignment model.
- `src/discriminator.py`: alignment loss helper.
- `src/train_gan.py`: training loop for alignment model.
- `src/eval_gan.py`: evaluation helpers.
- `src/datadownload.py`: dataset download/inspection.

Quickstart

```bash
cd "Ashraf - Project 5"

# Train alignment model
python src/train_gan.py --epochs 2 --batch_size 4
```

Notes

- Captions are derived from `<gdi imageX>` segments with grounding tags stripped.
- CLIP is frozen; only lightweight fusion, prefix, and heads are trained.

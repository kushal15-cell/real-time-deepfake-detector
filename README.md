## Model Improvement and Cross-Dataset Evaluation

The first EfficientNet-B0 model was trained primarily using face crops from the FaceForensics++ Deepfakes c23 dataset.

Although the baseline model performed well on FaceForensics++-style videos, it produced uncertain or incorrect predictions on external phone-camera and Celeb-DF videos.

This indicated a cross-dataset generalization problem: the model had learned dataset-specific visual patterns rather than universally transferable deepfake features.

### Baseline External Evaluation

The baseline model was evaluated using:

- 5 genuine real-world videos
- 5 external deepfake videos from Celeb-DF
- MTCNN face extraction
- One sampled frame per second
- Mean frame-probability aggregation
- Classification threshold of 0.50

| Metric | Baseline |
|---|---:|
| External accuracy | 70% |
| Genuine videos correctly classified | 4/5 |
| Deepfake videos correctly classified | 3/5 |
| Genuine-video false-positive rate | 20% |
| External fake recall | 60% |

### Mixed-Dataset Fine-Tuning

To improve generalization, the model was fine-tuned using a mixture of:

- FaceForensics++ genuine videos
- FaceForensics++ deepfake videos
- Celeb-DF genuine videos
- Celeb-DF deepfake videos

The mixed dataset contained:

| Split | Genuine faces | Fake faces |
|---|---:|---:|
| Training | 1,324 | 1,430 |
| Validation | 385 | 315 |

The best checkpoint was selected using validation loss.

| Best epoch | Train accuracy | Validation accuracy | Validation loss |
|---:|---:|---:|---:|
| 4 | 85.37% | 75.43% | 0.5260 |

Training accuracy continued increasing during epoch 5 while validation loss worsened, indicating the beginning of overfitting. Therefore, the epoch-4 checkpoint was retained.

### Improved External Evaluation

The improved model was evaluated using the same external test videos.

| Metric | Baseline | Improved |
|---|---:|---:|
| External accuracy | 70% | 80% |
| Genuine videos correctly classified | 4/5 | 5/5 |
| Deepfake videos correctly classified | 3/5 | 3/5 |
| Genuine-video false-positive rate | 20% | 0% |
| External fake recall | 60% | 60% |

Mixed-dataset fine-tuning improved external accuracy and eliminated false alarms on the tested genuine videos.

Deepfake recall remained unchanged, demonstrating that detecting unseen manipulation techniques remains difficult.

## Decision Policy

The application uses three result categories instead of presenting every score above 50% as definitive manipulation:

| Fake probability | Application result |
|---:|---|
| 0%–40% | Likely authentic |
| 40%–60% | Inconclusive |
| 60%–100% | Likely manipulated |

The uncertain region reduces misleading conclusions for borderline predictions.

## Limitations

- The external evaluation set contains only 10 videos and is not large enough to estimate production-level performance.
- Performance may degrade on unseen generators, heavy compression, occlusion, poor lighting, or small faces.
- Video predictions are produced by averaging sampled frame probabilities and do not currently model temporal relationships.
- Saliency overlays indicate influential image regions but do not prove which pixels were manipulated.
- The output should be treated as screening evidence rather than forensic proof.

## Conclusion

This project demonstrates an end-to-end deepfake detection workflow covering:

- video frame sampling
- MTCNN face detection
- EfficientNet-B0 fine-tuning
- frame-level and video-level inference
- explainability visualization
- Streamlit application development
- in-domain and external evaluation
- dataset-shift diagnosis
- mixed-dataset retraining
- baseline-versus-improved model comparison

The final mixed-dataset model improved external accuracy from 70% to 80% and reduced the tested genuine-video false-positive rate from 20% to 0%.
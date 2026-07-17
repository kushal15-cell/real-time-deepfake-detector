# Deepfake Detection Model Comparison

## Evaluation Setup

Both models were evaluated on the same external test set:

- 5 genuine real-world videos
- 5 Celeb-DF deepfake videos
- Video-level prediction using mean frame probability
- Decision threshold: 0.50

## Results

| Metric | Baseline Model | Mixed-Dataset Model |
|---|---:|---:|
| External accuracy | 70% | 80% |
| Real videos correctly classified | 4/5 | 5/5 |
| Fake videos correctly classified | 3/5 | 3/5 |
| Real false-positive rate | 20% | 0% |
| Fake recall | 60% | 60% |
| Best validation accuracy | Not recorded here | 75.43% |
| Best checkpoint epoch | 5 | 4 |

## Interpretation

The baseline EfficientNet-B0 model performed reasonably on FaceForensics++ data but showed weaker generalization on external videos.

After mixed-dataset fine-tuning using FaceForensics++ and Celeb-DF face crops:

- external accuracy improved from 70% to 80%
- the real-video false-positive rate dropped from 20% to 0%
- all five external genuine videos were classified correctly
- fake recall remained 60%, showing that unseen deepfake detection is still challenging

## Final Conclusion

Mixed-dataset fine-tuning improved generalization and reduced false alarms on genuine videos.

The detector should still be treated as a supporting screening system rather than forensic proof because some external deepfake videos were missed.
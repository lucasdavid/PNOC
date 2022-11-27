# Research Weakly Supervised

Studying regularization strategies for WSSS.
Experiments were run over LNCC SDumont infrastructure.

Many of the code lines here were borrowed from OC-CSE, Puzzle-CAM and CCAM repositories.

## Experiments

### VOC12

```shell
# 1. Train classifiers
sbatch runners/sdumont/1-classification/1-train-vanilla.sh
sbatch runners/sdumont/1-classification/1-train-puzzle.sh
sbatch runners/sdumont/1-classification/1-train-puzzle-occse.sh

sbatch runners/sdumont/1-classification/2-inference-classification.sh
```

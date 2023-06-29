# P-NOC: Adversarial CAM Generation for Weakly Supervised Semantic Segmentation

## Introduction

This respository contains the official implementation for the paper
"P-NOC: Adversarial CAM Generation for Weakly Supervised Semantic Segmentation".

In summary, P-NOC is trained by alternatively optimizing two objectives:
```math
\begin{align}
    \mathcal{L}_f &= \mathbb{E}_{(x,y)\sim\mathcal{D},r\sim y}[\mathcal{L}_\text{P} + \lambda_\text{cse}\ell_\text{cls}(p^\text{oc}, y\setminus\{r\})] \\
    \mathcal{L}_\text{noc} &= \mathbb{E}_{(x,y)\sim\mathcal{D},r\sim y}[\lambda_\text{noc}\ell_\text{cls}(p^\text{noc}, y)]
\end{align}
```
where $p^\text{noc} = oc(x \circ (1 - \psi(A^r) > \delta_\text{noc}))$.

![Diagram for the proposed P-NOC (Puzzle-Not so Ordinary Classifier) training setup.](assets/diagram-p-noc.png)

## Results
### Pascal VOC 2012 (test)

| Method | bg | a.plane | bike | bird  | boat  | bottle | bus   | car   | cat   | chair | cow   | d.table | dog   | horse | m.bike | person | p.plant | sheep | sofa  | train | tv | Overall |
| ---------- | ---------- | --------- | ------- | ----- | ----- | ------ | ----- | ----- | ----- | ----- | ----- | ----------- | ----- | ----- | --------- | ------ | ----------- | ----- | ----- | ----- | --------- | ------- |
| P-OC | 91.55      | 86.74     | 38.28   | 89.29 | 61.13 | 74.81  | 92.01 | 86.57 | 89.91 | 20.53 | 85.81 | 56.98       | 90.21 | 83.53 | 83.38     | 80.78  | 67.99       | 86.96 | 47.09 | 62.76 | 43.09     | 72.35   |
| P-NOC | 91.36      | 86.70     | 35.18   | 87.84 | 62.89 | 71.57  | 92.97 | 86.33 | 92.34 | 30.43 | 85.79 | 60.68       | 91.73 | 81.70 | 82.72     | 66.30  | 65.85       | 88.75 | 48.71 | 72.48 | 44.48     | 72.70   |

### MS COCO 2014 (val)

| Method | frisbee | skis | snowboard | sports ball | kite | baseball bat | baseball glove | skateboard | surfboard | tennis racket | bottle | wine glass | cup | fork | knife | spoon | bowl | banana | apple | sandwich | orange | broccoli | carrot | hot dog | pizza | donut | cake | chair | couch | potted plant | bed | dining table | toilet | tv | laptop | mouse | remote | keyboard | cell phone | microwave | oven | toaster | sink | refrigerator | book | clock | vase | scissors | teddy bear | hair drier | toothbrush | mIoU |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| P-NOC | 51.01 | 4.88 | 38.23 | 47.6 | 52.41 | 19.91 | 12.16 | 29.33 | 35.3 | 24.73 | 39.83 | 52.14 | 32.73 | 27.16 | 30.61 | 17.83 | 13.88 | 66.68 | 53.38 | 55.49 | 67.76 | 35.56 | 29.03 | 56.26 | 66.46 | 67.79 | 52.73 | 21.4 | 30.02 | 20.05 | 46.51 | 12.61 | 66.04 | 41.53 | 60.33 | 24.88 | 33.13 | 60.94 | 65.85 | 38.99 | 35.53 | 25.06 | 34.14 | 27.63 | 40.1 | 24.18 | 37.42 | 50.07 | 72.15 | 27.29 | 35.36 | 48.12 |

## Setup
Check the [SETUP.md](SETUP.md) file for information regarding the setup of the Pascal VOC 2012 and MS COCO 2014 datasets.

## Experiments

The scripts used for training P-NOC are available in the [runners](runners) folder.
Generally, they will run the following scripts, in this order:

```shell
./runners/0-setup.sh
./runners/1-priors.sh
./runners/2-saliency.sh
./runners/3-rw.sh
./runners/4-segmentation.sh
```

## Acknowledgements

Much of the code here was borrowed from psa, OC-CSE, Puzzle-CAM and CCAM repositories.
We thank the authors for their considerable contributions and efforts.

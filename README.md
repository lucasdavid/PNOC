# P-NOC: Adversarial CAM Generation for Weakly Supervised Semantic Segmentation

[![JVCI](https://img.shields.io/badge/paper-jvcir.2024.104187-green.svg)](https://doi.org/10.1016/j.jvcir.2024.104187) [![arXiv](https://img.shields.io/badge/arXiv-2305.12522-b31b1b.svg)](https://arxiv.org/abs/2305.12522)  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/lucasdavid/PNOC/blob/master/LICENSE)

## Introduction

This respository contains the official implementation for the paper "P-NOC: Adversarial Training of CAM Generating Networks for Robust Weakly Supervised Semantic Segmentation Priors".

![Diagram for the proposed training method P-NOC.](assets/diagram-noc-cse.png)

In summary, P-NOC is trained by alternatively optimizing two objectives:
```math
\begin{align}
    \mathcal{L}_f &= \mathbb{E}_{(x,y)\sim\mathcal{D},r\sim y}[\mathcal{L}_\text{P} + \lambda_\text{cse}\ell_\text{cls}(p^\text{oc}, y\setminus\{r\})] \\
    \mathcal{L}_\text{noc} &= \mathbb{E}_{(x,y)\sim\mathcal{D},r\sim y}[\lambda_\text{noc}\ell_\text{cls}(p^\text{noc}, y)]
\end{align}
```
where $`\mathcal{L}_\text{P}`$ is the Puzzle-CAM regularization and $`p^\text{noc} = oc(x \circ (1 - \psi(A^r) > \delta_\text{noc}))`$.

## Results
### Pascal VOC 2012 (test)

| Method | bg | a.plane | bike | bird  | boat  | bottle | bus   | car   | cat   | chair | cow   | d.table | dog   | horse | m.bike | person | p.plant | sheep | sofa  | train | tv | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P-OC | 91.6 | 86.7 | 38.3 | 89.3 | 61.1 | 74.8 | 92.0 | 86.6 | 89.9 | 20.5 | 85.8 | 57.0 | 90.2 | 83.5 | 83.4 | 80.8 | 68.0 | 87.0 | 47.1 | 62.8 | 43.1 | 72.4 |
| P-NOC | 91.7 | 87.9 | 38.1 | 80.9 | 66.1 | 69.8 | 93.8 | 86.4 | 93.2 | 37.4 | 83.6 | 60.9 | 92.3 | 84.7 | 83.8 | 80.5 | 62.3 | 81.9 | 53.1 | 77.7 | 36.7 | 73.5 |

### MS COCO 2014 (val)

| Method | bg | person | bicycle | car | motorcycle | airplane | bus | train | truck | boat | traffic light | fire hydrant | stop sign | parking meter | bench | bird | cat | dog | horse | sheep | cow | elephant | bear | zebra | giraffe | backpack | umbrella | handbag | tie | suitcase | frisbee | skis | snowboard | sports ball | kite | baseball bat | baseball glove | skateboard | surfboard | tennis racket | bottle | wine glass | cup | fork | knife | spoon | bowl | banana | apple | sandwich | orange | broccoli | carrot | hot dog | pizza | donut | cake | chair | couch | potted plant | bed | dining table | toilet | tv | laptop | mouse | remote | keyboard | cell phone | microwave | oven | toaster | sink | refrigerator | book | clock | vase | scissors | teddy bear | hair drier | toothbrush | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P-NOC | 81.8 | 55.1 | 55.3 | 47.4 | 70.3 | 56.3 | 76.8 | 68.4 | 54.6 | 49.0 | 46.6 | 77.4 | 74.4 | 71.5 | 40.4 | 62.3 | 76.5 | 76.1 | 68.1 | 75.3 | 78.5 | 80.6 | 85.0 | 80.7 | 73.6 | 28.0 | 63.3 | 14.4 | 15.5 | 54.1 | 50.4 | 8.2 | 42.7 | 54.5 | 46.3 | 19.1 | 14.2 | 26.5 | 34.9 | 20.0 | 40.0 | 42.7 | 36.2 | 23.2 | 27.8 | 17.3 | 16.6 | 62.9 | 53.3 | 46.4 | 62.1 | 41.1 | 28.4 | 55.1 | 62.7 | 66.4 | 54.3 | 25.2 | 34.3 | 25.4 | 44.5 | 13.7 | 65.1 | 40.7 | 55.9 | 23.2 | 30.0 | 60.1 | 65.5 | 46.4 | 36.2 | 36.5 | 34.4 | 27.7 | 37.9 | 25.3 | 35.8 | 54.1 | 71.8 | 29.1 | 37.3 | 47.7 |

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

## Artifacts and Intermediate Results

### Pascal VOC 2012

| #   | Method | Description | Train set |  dCRF | mIoU | Links |
| --- |    --- |         --- |       --- |   --- |  --- | ---   |
| —   | **CAMs** |
| 1   | vanilla+ra+ls      | priors | trainaug |  - | 53.7% | [weights](https://drive.google.com/file/d/1K5JPw_s1BZGxQuOBziy7Vne_90I0G9iR/view?usp=drive_link) [CAMs](https://drive.google.com/drive/folders/19RB0_3YBk63f4bUKcJrAleN-_ftqBpEk?usp=sharing) [wdb/train](https://wandb.ai/lerdl/research-wsss/runs/une4rz7c) wdb/eval
| 2   | P-OC (OC+ra)       | priors | trainaug |  - | 61.5% | [weights](https://drive.google.com/file/d/1K2ISPyZ9t3fdhjZq6A7lC1GAZHeNCwzc/view?usp=drive_link) [CAMs](https://drive.google.com/file/d/1mh3LNiAiUM-W4nFu7jJurbe_1BqSFwKv/view?usp=sharing) wdb/train [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/3c0ggzm8)
| 3   | P-OC+ls (OC+ra)    | priors | trainaug |  - | 61.9% | [weights](https://drive.google.com/file/d/1YWwwI9FZ0Jphs1g33NqkinkOT9Nt-VWO/view?usp=drive_link) [CAMs](https://drive.google.com/file/d/1qqwjdNcwR-XQihuZf1tXJ5vztGbcP4lS/view?usp=drive_link) wdb/train [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/1bsf0b47)
| 4   | P-NOC (OC+ra+ls)    | priors | trainaug |  - | 62.9% | [weights](https://drive.google.com/file/d/1vE_uxih236qgyp3arKVG5iEQPWOWVxt3/view?usp=drive_link) CAMs [wdb/train](https://wandb.ai/lerdl/research-wsss/runs/arnqbtp4) [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/arnqbtp4) |
| 5   | P-NOC+ls (OC+ra+ls) | priors | trainaug |  - | 63.7% | [weights](https://drive.google.com/file/d/1CDT89v8ctWOOowCgJkbh0Xj7fCxlQs2_/view?usp=drive_link) [CAMs](https://drive.google.com/file/d/1BPVs1z5FI5EdoqCIj1ygyyXVAZXBQaf7/view?usp=drive_link) [wdb/train](https://wandb.ai/lerdl/research-wsss/runs/mfjebixv) [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/iiurt3di) |
| —   | **Saliency** |
| 6   | C²AM-H (P-NOC+ls #5) | saliency | trainaug |  ✓ | 67.9% | [weights](https://drive.google.com/file/d/1R43ABBERuiL_p55OsRg0hm1yrp6eVQTz/view?usp=drive_link) [saliency](https://drive.google.com/file/d/1q_oPFAftc2gadWm0iZEFG-uO7RxTGWUN/view?usp=sharing) [wdb/train](https://wandb.ai/lerdl/research-wsss/runs/p47vbwfr) [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/u90rbym7) |
| 7   | PoolNet (C²AM-H  #6) | saliency | trainaug |  - | 70.8% | [weights](https://drive.google.com/file/d/13mzG1LY4gGQlUsRNegag73OrhiOEiyld/view?usp=drive_link) [saliency](https://drive.google.com/file/d/1WnLMR2LfYrW0fE6e1Z2hFvl62Q42PoBB/view?usp=drive_link) wdb/train [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/7lniygy8) |
| —   | **Random Walk** |
| 8   | AffinityNet (#5, #7) | affinity | trainaug | ✓ | -     | [masks](https://drive.google.com/file/d/1kobW99nDIee18xl0AeSo8IuMgOUu6KCZ/view?usp=drive_link) |
| 9   | AffinityNet (#5, #8) | pseudo masks | trainaug | ✓ | 75.5% | [weights](https://drive.google.com/file/d/11h9yoFV5pYdLu_IyrFkF30-RA-cMXKJk/view?usp=drive_link) [masks](https://drive.google.com/file/d/1NPiNpF5q0nXhvVr9JqN56tWlHz556gLH/view?usp=drive_link) wdb/train [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/cqlh4d5e) |
| —   | **Segmentation** |
| 10  | DeepLabV3+ (Supervised)      | segmentation | trainaug | ✓ | 80.6% | [weights](https://drive.google.com/file/d/1-ft7izarGC5fgSyV-f6QFJm75oMFfrZE/view?usp=sharing) [masks](https://drive.google.com/drive/folders/1kJVRHMpJpmfoCEh3NHpBI1NR2dRB9iTI?usp=sharing) [wdb/train](https://wandb.ai/lerdl/research-wsss/runs/1roq7b3b) [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/2d8qy9nr) |
| 11  | DeepLabV3+ (P-OC #2)         | segmentation | trainaug | ✓ | 71.4% | [weights](https://drive.google.com/file/d/1Lgr8kAoJ62MQzddEAoua2jsBh8VVIYMA/view?usp=sharing) [masks](https://drive.google.com/drive/folders/17FAHqb-P8KmHuOIwegx6IVYAB80KfnTT?usp=sharing) wdb/train [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/o4wrrd61) |
| 12  | DeepLabV3+ +ls (P-NOC+ls #7) | segmentation | trainaug | ✓ | 73.8% | [weights](https://drive.google.com/file/d/1WFz1f1E9f2xZpv380R0XXY_VDb_aoBuZ/view?usp=drive_link) [masks](https://drive.google.com/file/d/1-NcBDD8vPkFOTN8MN2Xq9m1FjKJBrEvE/view?usp=sharing) [wdb/train](https://wandb.ai/lerdl/research-wsss/runs/n7h2c3xu) [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/amb7bon7) |

### MS COCO 2014

| #   | Method | Description | Train set |  dCRF | mIoU (train) | Link |
| --- |                      --- |          --- |   --- | --- | --- |              --- |
| —   | **CAMs**                 |
| 1   | vanilla+ra               | priors       | train | - | - | [weights](https://drive.google.com/file/d/19WzA9QSAEFeJJI45UiOAZerB88W1lO8f/view?usp=sharing) CAMs |
| 2   | vanilla+ra+ls            | priors       | train | - | 33.7% | [weights](https://drive.google.com/file/d/1dK4UaxbBVA5DzIarPwXEO_nVwi_7qjBr/view?usp=sharing) CAMs [wdb/train](https://wandb.ai/lerdl/research-wsss/runs/b2n9ttz0) [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/yfjidtrk) |
| 3   | P-OC     (OC+ra #1)      | priors       | train | - | 38.5% | [weights](https://drive.google.com/file/d/1WXMVfLipvLE1a5XjqLDaUywfANmygFXH/view?usp=sharing) CAMs [wdb/train](https://wandb.ai/lerdl/research-wsss/runs/1ymousnh) [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/2j25yw2y) |
| 4   | P-OC+ls  (OC+ra+ls #2)   | priors       | train | - | 37.3% | [weights](https://drive.google.com/file/d/1SbffwA66UyG7T2dmDk1-526Yc9R_mGMq/view?usp=sharing) CAMs [wdb/train](https://wandb.ai/lerdl/research-wsss/runs/268yxg3s) [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/7c31u0qf) |
| 5   | P-NOC (OC+ra #1)         | priors       | train | - | 40.7% | [weights](https://drive.google.com/file/d/1JKfO6ZZ2b3maIKMUs_y641OA0ouRReEt/view?usp=sharing) CAMs [wdb/train](https://wandb.ai/lerdl/research-wsss/runs/xn6e8z6n) [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/22u4104t) |
| 6   | P-NOC+ls (OC: RS269+ra)  | priors       | train | - | 38.2% | [weights](https://drive.google.com/file/d/1omm8tRhR-Zdl7NrgSmGo3RdYcLZIoIg_/view?usp=drive_link) CAMs [wdb/train](https://wandb.ai/lerdl/research-wsss/runs/3h67i1np) [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/e4kq1ntv) |
| —   | **Saliency** |
| 6   | C²AM-H (P-NOC #5)        | saliency    | trainaug |  ✓ | 70.5% | [weights](https://drive.google.com/file/d/1nbiDk0zAseddqlcE6SfBgmRF3Do_kTjN/view?usp=sharing) saliency [wdb/train](https://wandb.ai/lerdl/research-wsss/runs/1u9nx4pl) [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/9km9e6ko) |
| 7   | PoolNet (C²AM-H #7)      | saliency    | trainaug |  - | 71.3% | [weights](https://drive.google.com/file/d/19R_S0dAWGWmLalQeHFnj2by_ekVCUuK4/view?usp=sharing) [saliency](https://drive.google.com/file/d/1O8wa7HTN90c3KAEnG2jqVlrVcy2oBbBn/view?usp=sharing) wdb/train [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/26wir4o1) |
| —   | **Random Walk**          |
| 8   | AffinityNet (#5, #7)     | affinity     | train | ✓ | -     | [masks](https://drive.google.com/file/d/1Whwd4Y7vLzOqhTJYTJuGPjtZsThEDzow/view?usp=sharing)             |
| 9   | AffinityNet (#5, #7, #8) | pseudo masks | train | ✓ | 47.7% | [weights](https://drive.google.com/file/d/13NWIdguVEgcvrNnPQ5jio3c9WuB-6tA8/view?usp=sharing) [masks](https://drive.google.com/file/d/1fvB_w_tNcVFQC0AHdpOPRBxC_Yp-UOTJ/view?usp=drive_link) [wdb/train](https://wandb.ai/lerdl/research-wsss/runs/1r57g8pe) [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/2foqmfc2) |
| —   | **Segmentation**         |
| 2   | DeepLabV3+ (P-NOC #2)    | segmentation | train | - | 44.6% | [weights](https://drive.google.com/file/d/16v95VqTgyx_4MiI5uNBdPf6_s4ih6yYv/view?usp=sharing) [masks](https://drive.google.com/file/d/1HZsZBpvY5gT9GD_x0Fk8lHbaKKurrzcH/view?usp=sharing) [wdb/train](https://wandb.ai/lerdl/research-wsss/runs/igyhotem) [wdb/eval](https://wandb.ai/lerdl/research-wsss/runs/g0al44je) |

## Citation

If our work was helpful to you, please cite it as:

```
@article{david2024104187pnoc,
title = {P-NOC: Adversarial training of CAM generating networks for robust weakly supervised semantic segmentation priors},
journal = {Journal of Visual Communication and Image Representation},
volume = {102},
pages = {104187},
year = {2024},
issn = {1047-3203},
doi = {https://doi.org/10.1016/j.jvcir.2024.104187},
author = {Lucas David and Helio Pedrini and Zanoni Dias}
```

## Acknowledgements

Much of the code here was borrowed from [jiwoon-ahn/psa](https://github.com/jiwoon-ahn/psa), [KAIST-vilab/OC-CSE](https://github.com/KAIST-vilab/OC-CSE), [shjo-april/PuzzleCAM](https://github.com/shjo-april/PuzzleCAM) and [CVI-SZU/CCAM](https://github.com/CVI-SZU/CCAM) repositories.
We thank the authors for their considerable contributions and efforts.


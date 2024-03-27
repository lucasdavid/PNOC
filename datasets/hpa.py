"""
HPA - Single Cell Classification

Label 	Name 	Description
 0. 	Nucleoplasm 	The nucleus is found in the center of cell and can be identified with the help of the signal in the blue nucleus channel. A staining of the nucleoplasm may include the whole nucleus or of the nucleus without the regions known as nucleoli (Class 2).
 1. 	Nuclear membrane 	The nuclear membrane appears as a thin circle around the nucleus. It is not perfectly smooth and sometimes it is also possible to see the folds of the membrane as small circles or dots inside the nucleus.
 2. 	Nucleoli 	Nucleoli can be seen as slightly elongated circular areas in the nucleoplasm, which usually display a much weaker staining in the blue DAPI channel. The number and size of nucleoli varies between cell types.
 3. 	Nucleoli fibrillar center 	Nucleoli fibrillary center can appear as a spotty cluster or as a single bigger spot in the nucleolus, depending on the cell type.
 4. 	Nuclear speckles 	Nuclear speckles can be seen as irregular and mottled spots inside the nucleoplasm.
 5. 	Nuclear bodies 	Nuclear bodies are visible as distinct spots in the nucleoplasm. They vary in shape, size and numbers depending on the type of bodies as well as cell type, but are usually more rounded compared to nuclear speckles.
 6. 	Endoplasmic reticulum 	The endoplasmic reticulum (ER) is recognized by a network-like staining in the cytosol, which is usually stronger close to the nucleus and weaker close to the edges of the cell. The ER can be identified with the help of the staining in the yellow ER channel.
 7. 	Golgi apparatus 	The Golgi apparatus is a rather large organelle that is located next to the nucleus, close to the centrosome, from which the microtubules in the red channel originate. It has a folded ribbon-like appearance, but the shape and size can vary between cell types, and in response to cellular various processes.
 8. 	Intermediate filaments 	Intermediate filaments often exhibit a slightly tangled structure with strands crossing every so often. They can appear similar to microtubules, but do not match well with the staining in the red microtubule channel. Intermediate filaments may extend through the whole cytosol, or be concentrated in an area close to the nucleus.
 9. 	Actin filaments 	Actin filaments can be seen as long and rather straight bundles of filaments or as branched networks of thinner filaments. They are usually located close to the edges of the cells.
10. 	Microtubules 	Microtubules are seen as thin strands that stretch throughout the whole cell. It is almost always possible to detect the center from which they all originate (the centrosome). And yes, as you might have guessed, this overlaps the staining in the red channel.
11. 	Mitotic spindle 	The mitotic spindle can be seen as an intricate structure of microtubules radiating from each of the centrosomes at opposite ends of a dividing cell (mitosis). At this stage, the chromatin of the cell is condensed, as visible by intense DAPI staining. The size and exact shape of the mitotic spindle changes during mitotic progression, clearly reflecting the different stages of mitosis.
12. 	Centrosome 	This class includes centrosomes and centriolar satellites. They can be seen as a more or less distinct staining of a small area at the origin of the microtubules, close to the nucleus. When a cell is dividing, the two centrosomes move to opposite ends of the cell and form the poles of the mitotic spindle.
13. 	Plasma membrane 	This class includes plasma membrane and cell junctions. Both are at the outer edge of the cell. Plasma membrane sometimes appears as a more or less distinct edge around the cell, occasionally with characteristic protrusions or ruffles. In some cell lines, the staining can be uniform across the entire cell. Cell junctions can be observed at contact sites between neighboring cells.
14. 	Mitochondria 	Mitochondria are small rod-like units in the cytosol, which are often distributed in a thread-like pattern along microtubules.
15. 	Aggresome 	An aggresome can be seen as a dense cytoplasmic inclusion, which is usually found close to the nucleus, in a region where the microtubule network is disrupted.
16. 	Cytosol 	The cytosol extends from the plasma membrane to the nuclear membrane. It can appear smooth or granular, and the staining is often stronger close to the nucleus.
17. 	Vesicles and punctate cytosolic patterns 	This class includes small circular compartments in the cytosol: Vesicles, Peroxisomes (lipid metabolism), Endosomes (sorting compartments), Lysosomes (degradation of molecules or eating up dead molecules), Lipid droplets (fat storage), Cytoplasmic bodies (distinct granules in the cytosol). They are highly dynamic, varying in numbers and size in response to environmental and cellular cues. They can be round or more elongated.
18. 	Negative 	This class include negative stainings and unspecific patterns. This means that the cells have no green staining (negative), or have staining but no pattern can be deciphered from the staining (unspecific).
"""

import os
import cv2
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split

from . import base

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "hpa-single-cell-classification")
CLASSES = [
  "background", "Nucleoplasm", "Nuclearmembrane", "Nucleoli", "Nucleolifibrillar", "Nuclearspeckles", "Nuclearbodies", "Endoplasmic",
  "Golgi", "Intermediate", "Actin", "Microtubules", "Mitotic", "Centrosome", "Plasma",
  "Mitochondria", "Aggresome", "Cytosol", "Vesicles", "Negative",
]
COLORS = [
  [0, 0, 0], [172, 47, 117], [192, 67, 195], [103, 9, 211], [21, 36, 87], [70, 216, 88], [140, 58, 193], [39, 87, 174],
  [88, 81, 165], [25, 77, 72], [9, 148, 115], [208, 197, 79], [175, 192, 82], [99, 216, 177], [29, 147, 147],
  [142, 167, 32], [193, 9, 185], [127, 32, 31], [202, 151, 163], [203, 114, 183], [224, 224, 192],
]


class HPASingleCellClassificationDataSource(base.CustomDataSource):
  NAME = "hpa-single-cell-classification"
  VALIDATION_SPLIT = 0.1
  SEED = 1838339744

  def __init__(
    self,
    root_dir,
    domain: str,
    split: Optional[str] = "train",
    images_dir=None,
    masks_dir: str = None,
    sample_ids: List[str] = None,
  ):
    images_dir = images_dir or os.path.join(root_dir, domain or "train")
    masks_dir = masks_dir or os.path.join(root_dir, "SegmentationClass")
    self.root_dir = root_dir

    super().__init__(
      domain=domain,
      split=split,
      images_dir=images_dir,
      masks_dir=masks_dir,
      sample_ids=sample_ids,
    )
    self.sample_labels = self.load_sample_labels(self.domain)

  _sample_info: Dict[str, Tuple[np.ndarray, np.ndarray]] = None  # {train: (<ids shape=N dtype=str>, <labels shape=(N, 19) dtype=float>)}

  def load_sample_info(self, domain: str):
    if self._sample_info is None:
      train_info = pd.read_csv(os.path.join(self.root_dir, "train.csv"))
      with open(os.path.join(self.root_dir, "sample_submission.csv")) as f:
        test_ids = np.asarray([l.strip().split(",")[0] for l in f.readlines()])

      train_targets = np.zeros((len(train_info), 19))
      test_targets = np.zeros((len(test_ids), 19))
      for i, l in enumerate(train_info["Label"]):
        train_targets[i, list(map(int, l.split("|")))] = 1.

      train_ids, valid_ids, train_targets, valid_targets = train_test_split(
        train_info.ID.values, train_targets, test_size=self.VALIDATION_SPLIT, random_state=self.SEED)

      self._sample_info = {
        "train": (train_ids, train_targets),
        "valid": (valid_ids, valid_targets),
        "test": (test_ids, test_targets),
      }

    return self._sample_info[domain]

  def get_sample_ids(self, domain) -> List[str]:
    return self.load_sample_info(domain)[0]

  def load_sample_labels(self, domain) -> List[str]:
    ids, targets = self.load_sample_info(domain)
    return dict(zip(ids, targets))
  
  def get_image(self, sample_id) -> Image.Image:
    colors = ('red','green','blue','yellow')
    image = [cv2.imread(os.path.join(self.root_dir, self.domain, f'{sample_id}_{c}.png'), cv2.IMREAD_GRAYSCALE) for c in colors]
    image = np.stack(image, axis=-1)
    image = Image.fromarray(image)
    return image

  def get_label(self, sample_id: str) -> np.ndarray:
    label = self.sample_labels[sample_id]
    return label

  def get_info(self, task: str) -> base.DatasetInfo:
    if task == "segmentation":
      num_classes = 20
      classes = CLASSES
      colors = COLORS
      bg_class = 0
      void_class = 20
    else:
      # without bg and void:
      num_classes = 19
      classes = CLASSES[1:]
      colors = COLORS[1:]
      bg_class = None
      void_class = None

    return base.DatasetInfo(
      num_classes=num_classes,
      channels=4,
      classes=classes,
      colors=colors,
      bg_class=bg_class,
      void_class=void_class,
    )


base.DATASOURCES[HPASingleCellClassificationDataSource.NAME] = HPASingleCellClassificationDataSource

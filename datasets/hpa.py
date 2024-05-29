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
NORMALIZE_STATS = (
  (0.485, 0.456, 0.406, 0.485),
  (0.229, 0.224, 0.225, 0.229))

class HPASingleCellClassificationDataSource(base.CustomDataSource):
  NAME = "hpa-single-cell-classification"
  DEFAULT_SPLIT = "train"
  DOMAINS = {
    "train": "train_aug",
    "valid": "valid",
    "test": "test",
  }

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
    masks_dir = masks_dir or os.path.join(root_dir, "cell_masks")
    self.root_dir = root_dir

    super().__init__(
      domain=domain,
      split=split,
      images_dir=images_dir,
      masks_dir=masks_dir,
      sample_ids=sample_ids,
    )
    self.sample_labels = self.load_sample_labels(self.domain)
    self.subfolder = "test" if self.domain == "test" else "train"
    self.masks_format = "npz"

  _sample_info: Dict[str, Tuple[np.ndarray, np.ndarray]] = None  # {train: (<ids shape=N dtype=str>, <labels shape=(N, 19) dtype=float>)}

  def load_sample_info(self, domain: str):
    if self._sample_info is None:
      train_info = pd.read_csv(os.path.join(self.root_dir, "train.csv"))
      public_info = pd.read_csv(os.path.join(self.root_dir, "publichpa.csv"))
      with open(os.path.join(self.root_dir, "sample_submission.csv")) as f:
        ids_test = np.asarray([l.strip().split(",")[0] for l in f.readlines()])

      y_train = np.zeros((len(train_info), 19))
      for i, l in enumerate(train_info["Label"]):
        y_train[i, list(map(int, l.split("|")))] = 1.

      y_pub = np.zeros((len(public_info), 19))
      for i, l in enumerate(public_info["Label"]):
        y_pub[i, list(map(int, l.split("|")))] = 1.

      y_test = np.zeros((len(ids_test), 19))

      ids_train = train_info.ID.values
      ids_pub = public_info.ID.values
      ids_train, ids_val, y_train, y_val = train_test_split(
        ids_train, y_train, test_size=self.VALIDATION_SPLIT, random_state=self.SEED)

      cat = lambda *x: np.concatenate(x, 0)

      self._sample_info = {
        "train": (ids_train, y_train),
        "valid": (ids_val, y_val),
        "train_val": (cat(ids_train, ids_val), cat(y_train, y_val)),
        "train_aug": (cat(ids_train, ids_pub), cat(y_train, y_pub)),
        "train_aug_val": (cat(ids_train, ids_pub, ids_val), cat(y_train, y_pub, y_val)),
        "test": (ids_test, y_test),
      }

    return self._sample_info[domain]

  def get_sample_ids(self, domain) -> List[str]:
    return self.load_sample_info(domain)[0]

  def load_sample_labels(self, domain) -> List[str]:
    ids, targets = self.load_sample_info(domain)
    return dict(zip(ids, targets))

  def get_image(self, sample_id) -> Image.Image:
    colors = ('red','green','blue','yellow')
    images = [Image.open(os.path.join(self.root_dir, self.subfolder, f'{sample_id}_{c}.png')) for c in colors]
    image = np.stack([np.array(image) for image in images], axis=-1)
    image = Image.fromarray(image)

    for i in images: i.close()

    return image

  def get_mask_path(self, sample_id) -> str:
    return os.path.join(self.masks_dir, sample_id + '.npz')

  def get_mask(self, sample_id):
    mask_path = self.get_mask_path(sample_id)
    if os.path.exists(mask_path):
      mask = np.load(mask_path)["arr_0"]
      mask = Image.fromarray(mask)
      return mask
    else:
      return Image.fromarray(np.full((32, 32), 255, dtype="uint8"))

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
      normalize_stats=NORMALIZE_STATS,
    )


base.DATASOURCES[HPASingleCellClassificationDataSource.NAME] = HPASingleCellClassificationDataSource

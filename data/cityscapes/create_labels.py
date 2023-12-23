import sys

import numpy as np
from datasets import cityscapes


def main():
  train_source = cityscapes.CityscapesDataSource("./data/cityscapes", "train")
  valid_source = cityscapes.CityscapesDataSource("./data/cityscapes", "val")
  test_source  = cityscapes.CityscapesDataSource("./data/cityscapes", "test")

  for tag, source in (("train", train_source), ("val", valid_source), ("test", test_source)):
    labels = [np.unique(np.asarray(source.get_mask(_id))) for _id in source.sample_ids]
    lines = [
      ",".join((_id, "|".join(label[label != 255].astype(str).tolist()))) + "\n"
      for _id, label in zip(source.sample_ids, labels)
    ]

    with open(f"../data/cityscapes/{tag}.txt", "w") as f:
      f.writelines(lines)


if __name__ == "__main__":
  main()

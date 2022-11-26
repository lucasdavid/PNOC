import torch
import torchvision.transforms as transforms
from core.datasets import *
from core.mcar import mcar_resnet101
from tools.ai.augment_utils import *

# Model
ps = 'avg'
topN = 4
threshold = 0.5
model = mcar_resnet101(
  20,
  ps,
  topN,
  threshold,
  inference_mode=True,
)

ckpt_file = '/home/ldavid/workspace/logs/sdumont/mcar/model_best.pth.tar'
ckpt = torch.load(ckpt_file, map_location=torch.device('cpu'))
model.load_state_dict(ckpt['state_dict'], strict=True)
model.eval()

# Dataset
classes = np.asarray([
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable",
    "dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
])


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

meta_dic = read_json('./data/voc12/meta.json')

import torchmetrics
from core.datasets import *
from tools.ai.augment_utils import *
from torch.utils.data import DataLoader

val_dataset = VOC12ClassificationDataset(
  '/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/', 'val',
  transforms.Compose([Normalize(imagenet_mean, imagenet_std),
                      Top_Left_Crop(512), Transpose()])
)

val_loader = DataLoader(val_dataset, batch_size=8)
steps = len(val_loader)

# Metrics
metrics = {'f1': torchmetrics.F1Score(num_classes=20, average='none')}

for step, (x, y) in enumerate(val_loader):
  p, = model(x)

  metrics['f1'](p, y.int())

  if (step + 1) % (steps // 10) == 0:
    f1 = metrics['f1'].compute()
    print(f'F1 on step {step / steps:.0%}: {f1}')

# metric on all batches using custom accumulation
f1 = metrics['f1']
f1_values = f1.compute().detach().numpy()
print(*(f'{c:<12}: {v:.2%}' for c, v in zip(classes, f1_values)), sep='\n')

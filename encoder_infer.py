import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from torch.autograd import Variable

from dataset import get_loader
from evaluate import run_eval
from train_options import parser
from util import get_models, init_lstm, set_train, set_eval
from util import prepare_inputs, forward_ctx 


args = parser.parse_args()
print(args) 

#####1. Data ####
## using the same loader as of train.py 
# this loader can be extended to evaluate over multiple datasets
def get_eval_loaders():
  # We can extend this dict to evaluate on multiple datasets.
  eval_loaders = {'TVL': get_loader(is_train=False, root=args.eval, mv_dir=args.eval_mv, args=args),}
  return eval_loaders

data_loader = get_eval_loaders()

#####2. Encoder Model #### 
encoder, binarizer, _, unet = get_models(
    args=args, v_compress=args.v_compress, 
    bits=args.bits,
    encoder_fuse_level=args.encoder_fuse_level,
    decoder_fuse_level=args.decoder_fuse_level)

nets = [encoder, binarizer]
if unet is not None:
    nets.append(unet)


####3. GPU's setup #### 
gpus = [int(gpu) for gpu in args.gpus.split(',')]
if len(gpus) > 1:
  print("Using GPUs {}.".format(gpus))
  for net in nets:
    net = nn.DataParallel(net, device_ids=gpus)

####4.load the saved parameters ####  

## check for model directory
if not os.path.exists(args.model_dir):
  print("model directory not present. check the train.py and create it during training to save weights to")

def load_weights():
  names = ['encoder', 'binarizer','unet']
  print("loading  weights for ", names)

  for net_idx, net in enumerate(nets):
    if net is not None:
      name = names[net_idx]
      load_path = '{}/{}.pth'.format(args.model_dir, name) 
      net.load_state_dict(torch.load(load_path))

  print("done.")

load_weights() 

####5. forward the model and save code  
set_eval(nets)

for name, loader in data_loader.items():
  run_eval(nets, loader, args)
  

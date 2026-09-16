#!/usr/bin/env python3
"""Retrain the public Bearing-UAV architecture on our selected Route-A only.

Unlike bearinguav_official_route_eval.py (which uses the authors' full-dataset
pretrained checkpoint), this script trains the official PARCASGM_v5a/VGG16
architecture only from the selected train_01 rows for one city, then evaluates
exactly the selected test_01/test_02 rows.

The public architecture, position+heading targets, SmoothL1 objective and
0.8/0.2 loss weighting are retained.  Image decoding is cached once into uint8
.npy tensors; later epochs mmap those tensors and augment/normalize on GPU, so
CPU workers do not repeatedly decode five images per sample.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bearing_prepare as bearing
from bearinuav_dummy import dummy  # type: ignore  # replaced immediately below

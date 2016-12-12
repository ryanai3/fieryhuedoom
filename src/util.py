import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import cuda
import math

def overlap(x1, w1, x2, w2):
  l1 = x1 - w1/2
  l2 = x2 - w2/2
  left = l1 if l1 > l2 else l2
  
  r1 = x1 + w1/2
  r2 = x2 + w2/2
  right = r1 if r1 < r2 else r2
  return max(right - left, 0)

def box_intersection(box_a, box_b):
  w = overlap(box_a[0], box_a[2], box_b[0], box_b[2])
  h = overlap(box_a[1], box_a[3], box_b[1], box_b[3])
  if w == 0 or h == 0:
    return 0
  return h * w

def box_iou(box_a, box_b):
  i = box_intersection(box_a, box_b)
  u = (box_a[2] * box_a[3]) + (box_b[2] * box_b[3]) - i
  return i / u

def box_rmse(box_a, box_b):
  return math.sqrt(sum((box_a - box_b) ** 2))

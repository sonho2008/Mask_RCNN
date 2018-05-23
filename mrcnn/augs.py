

import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL


def aug_rotate(*args):
  return #TODO

def aug_flipud(*args):
  return #TODO

def aug_fliplr(*args):
  return #TODO

def get_augmenter(name, args):
  return globals()['aug_' + name](*args)

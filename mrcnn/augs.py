

import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL

def bbox_from_mask(mask):
    return #TODO

def aug_rotate(*args):
    return #TODO

def aug_flipud(*args):
    return #TODO

def flip_lr_func(image):
    imshape = list(image.get_shape())
    if len(imshape)==3: #[H,W,3]
        return tf.image.flip_left_right(image)
    elif len(imshape)==4: #[bsize, H, W, 3]
        images = tf.split(image, num_or_size_splits=imshape[0], axis=0)
        images_3d = [tf.reshape(im, shape=imshape[1:]) for im in images]
        flipped_ims = [tf.reshape(flip_lr_func(im), shape=[1]+imshape[1:])
            for im in images_3d]
        return tf.concat(flipped_ims, axis=0)
    else:
        raise ValueError("unexpected image shape")

def aug_fliplr(*args):
    #TODO : randomize using args
    layer = KL.Lambda(flip_lr_func)
    return layer

def get_augmenter(name, args):
    return globals()['aug_' + name](*args)

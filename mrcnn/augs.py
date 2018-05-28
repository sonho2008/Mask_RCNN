

import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
from math import pi

def bbox_from_mask(mask):
    return #TODO

def random_choice(t1, t2, prob1=0.5):
    c = tf.constant(prob1)
    r = tf.random_uniform([], 0.0, 1.0)
    return tf.cond(r<c, lambda:t1, lambda:t2)

def flip_func(image, flip_op=tf.image.flip_left_right):
    imshape = list(image.get_shape())
    if len(imshape)==3: #[H,W,3]
        return flip_op(image)
    elif len(imshape)==4: #[bsize, H, W, 3]
        images_3d = tf.unstack(image, axis=0)
        flipped_ims = [tf.reshape(flip_func(im, flip_op), shape=[1]+imshape[1:])
            for im in images_3d]
        return tf.concat(flipped_ims, axis=0)
    else:
        raise ValueError("unexpected image shape")

def rotate_func(images, angle_range):
    angle = tf.random_uniform([], angle_range[0], angle_range[1])
    radian = angle / (pi*180.0)
    return tf.contrib.image.rotate(images, radian, "BILINEAR")

def aug_fliplr(prob=0.5):
    return KL.Lambda(lambda image: random_choice(
        flip_func(image, tf.image.flip_left_right), image, prob)
    )

def aug_flipud(prob=0.5):
    return KL.Lambda(lambda image: random_choice(
        flip_func(image, tf.image.flip_up_down), image, prob)
    )

def aug_rotate(angle_range=(-180, 180), prob=1.0):
    return KL.Lambda(lambda image: random_choice(
        rotate_func(image, angle_range), image, prob)
    )

def get_augmenter(name, *args, **kwargs):
    return globals().get('aug_' + name)(*args, **kwargs)


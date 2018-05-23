

import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL

def bbox_from_mask(mask):
    return #TODO

def random_choice(t1, t2, prob1=0.5):
    c = tf.constant(prob1)
    r = tf.random_uniform([], 0.0, 1.0)
    return tf.cond(r<c, lambda:t1, lambda:t2)

def flip_lr_func(image):
    imshape = list(image.get_shape())
    if len(imshape)==3: #[H,W,3]
        return tf.image.flip_left_right(image)
    elif len(imshape)==4: #[bsize, H, W, 3]
        images_3d = tf.unstack(image, axis=0)
        flipped_ims = [tf.reshape(flip_lr_func(im), shape=[1]+imshape[1:])
            for im in images_3d]
        return tf.concat(flipped_ims, axis=0)
    else:
        raise ValueError("unexpected image shape")

def rotate_func(image):
    imshape = list(image.get_shape())
    return #TODO

def aug_fliplr(prob=0.5):
    layer = KL.Lambda(lambda image: random_choice(flip_lr_func(image), image, prob))
    return layer

def aug_rotate(*args):
    return #TODO

def aug_flipud(*args):
    return #TODO

def get_augmenter(name, args):
    return globals()['aug_' + name](*args)


import tensorflow as tf
import numpy as np
import model as m
import skimage.io 
import keras
x=skimage.io.imread(fname='/media/ltp/40BC89ECBC89DD32/LIPHP_data/LIP/SinglePerson/parsing/train_segmentations/'+'839_1226048.png')
im_shape=x.shape
x=tf.one_hot(tf.expand_dims(x,0),20,1.0,0.0)
c=m.ComputeCentersLayer(num_classes=20)(x)
c._keras_shape=tf.shape(c)

t=m.GRenderLayer(1,im_shape[:2])(c)
sess=tf.Session()
result = sess.run([x,c,t])
import matplotlib.pyplot as plt
skimage.io.imshow(np.sum(result[2][0],-1))
plt.show()
#inputs=[tf.constant(11),tf.constant(12),tf.constant(13)]
#x=keras.layers.Lambda(lambda x:m.random_choice_graph(x))(inputs)
#y=keras.backend.random_uniform(shape=(),minval=0,maxval=3,dtype=tf.int32,seed=0)
#sess=tf.Session()
#for v in range(20):
    #print(sess.run([x,y]))
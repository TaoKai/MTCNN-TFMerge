import tensorflow as tf

sess = tf.Session()
stop = tf.placeholder(tf.int32, None)
i0 = tf.constant(0)
m0 = tf.ones([1, 5])
c = lambda i, m: i < stop

def body(i, m):
    i += 1
    m = tf.concat([m, m0], axis=0)
    return i, m

r = tf.while_loop(c, body, loop_vars=[i0, m0], shape_invariants=[i0.get_shape(), tf.TensorShape([None, 5])])
# ret = sess.run(r, {stop:9})
# print(ret[1], ret[1].shape)

nums = tf.range(25)
nums = tf.reshape(nums, [5, 5, 1])
nums = tf.tile(nums, [1, 1, 3])
inds = tf.constant([[1, 1], [3, 3], [2, 4], [4, 4]])
more_inds = tf.stack([inds, inds], axis=0)
vecs = tf.gather_nd(nums, more_inds)
ret = sess.run(more_inds)
print(ret, ret.shape)
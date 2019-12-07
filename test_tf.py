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
sess.run(tf.global_variables_initializer())
ret = sess.run(r, {stop:9})
print(ret[1], ret[1].shape)
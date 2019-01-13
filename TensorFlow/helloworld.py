import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)

add = tf.add(5,2)
print("Addition (5,2):",add)

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(x, feed_dict={x: 'Test String', y: 123})
    print(output)
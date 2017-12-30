import  tensorflow as tf

mat1 = tf.constant([[3, 3]])
mat2 = tf.constant([[2],
                    [2]])
product = tf.matmul(mat1, mat2)

# method 1
sess1 = tf.Session()
res1 = sess1.run(product)
print("method1: ", res1)
sess1.close()

# method 2
with tf.Session() as sess2:
    res2 = sess2.run(product)
    print("method2: ", res2)


import tensorflow as tf
slim=tf.contrib.slim
class siamese:

    # Create model
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, 50,50,1])
        self.x2 = tf.placeholder(tf.float32, [None, 50,50,1])

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.loss_with_spring()
        self.acc=self.test()
    def network(self, x):
        weights = []
        conv1=slim.conv2d(x, 64,[3,3], scope='conv1')
        conv2=slim.conv2d(conv1, 32, [3,3], scope='conv2')
        conv3=slim.conv2d(conv2, 16, [3,3], scope='conv3')
        flat=slim.flatten(conv3)
        fc1=slim.fully_connected(flat, 2048, scope='fc1')
        fc2=slim.fully_connected(fc1, 256, scope='fc2')
        fc3 = slim.fully_connected(fc2, 2,activation_fn=tf.nn.sigmoid,scope='fc3')
        return fc3


    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)#L2,
        eucd2 = tf.reduce_sum(eucd2, 1)#均方差
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def loss_with_step(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
        neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss
    def test(self):
        margin = 5.0
        labels = self.y_
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2 + 1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        result=eucd<C
        result=tf.cast(result,'float')
        correct_prediction = tf.equal(result,labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return  accuracy

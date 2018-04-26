import tensorflow as tf
import numpy as np
from local_linear_explanation import LocalLinearExplanation
import time
import tensorflow.contrib.slim as slim


def to_logprob(L): return L - tf.reduce_logsumexp(L, axis=1, keep_dims=True)


def one_hot(y):
    if len(y.shape) != 1:
        return y
    values = np.array(sorted(list(set(y))))
    return np.array([values == v for v in y], dtype=np.uint8)


class TensorflowCNN():

    def CNN(self,inputs, is_training=True):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            x = inputs #tf.reshape(inputs, [-1, 28, 28, 1])
            print(x)
            net = slim.conv2d(x, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.flatten(net, scope='flatten3')
            net = slim.fully_connected(net, 1600, scope='fc3')
            net = slim.dropout(net, is_training=is_training, scope='dropout3')
            outputs = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
        return outputs



    def loss_function(self, l2_grads=1000, l1_grads=0, l2_params=0.0001):
        right_answer_loss = tf.reduce_sum(
            tf.multiply(self.y, -self.log_prob_ys))

        gradXes = tf.gradients(self.log_prob_ys, self.X)[0]
        A_gradX = tf.multiply(self.A, gradXes)
        right_reason_loss = 0
        if l1_grads > 0:
            right_reason_loss += l1_grads * tf.reduce_sum(tf.abs(A_gradX))
        if l2_grads > 0:
            right_reason_loss += l2_grads * tf.nn.l2_loss(A_gradX)

        small_params_loss = l2_params * \
            tf.add_n([tf.nn.l2_loss(p) for p in self.W + self.b])

        return right_answer_loss + right_reason_loss + small_params_loss

    def optimizer(self, l2_grads=1000, l1_grads=0, l2_params=0.0001, learning_rate=0.001):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(self.loss_function(l2_grads=l2_grads, l1_grads=l1_grads, l2_params=l2_params))

    def fit(self, X, y, A=None,
            hidden_layers=[50, 30], nonlinearity=tf.nn.relu, always_include=None,weight_sd=0.1,
            l2_grads=1000, l1_grads=0, l2_params=0.001,
            num_epochs=64, batch_size=256, learning_rate=0.001):

        y = one_hot(y)
        y_dimensions = y.shape[1]
        x_dimensions = X.shape[1:]
        num_examples = X.shape[0]
        if A is None:
            #A = np.zeros((num_examples, x_dimensions))
            A = np.zeros_like(X)

        assert(num_examples == y.shape[0])
        assert(A.shape == X.shape)

        # set up network
        """
        self.layer_sizes = [x_dimensions] + \
            list(hidden_layers) + [y_dimensions]
        self.X = tf.placeholder("float", [None, x_dimensions], name="X")
        self.A = tf.placeholder("float", [None, x_dimensions], name="A")
        self.y = tf.placeholder("float", [None, y_dimensions], name="y")
        self.W = []
        self.b = []
        self.L = [self.X]
        for i in range(1, len(self.layer_sizes)):
            self.W.append(tf.Variable(tf.random_normal(
                self.layer_sizes[i-1:i+1], stddev=weight_sd), name='W{}'.format(i)))
            self.b.append(tf.Variable(tf.random_normal(
                [self.layer_sizes[i]], stddev=weight_sd), name='b{}'.format(i)))
        for i, activation in enumerate([nonlinearity for _ in hidden_layers] + [to_logprob]):
            self.L.append(activation(
                tf.add(tf.matmul(self.L[i], self.W[i]), self.b[i])))

        """    
        self.X = tf.placeholder("float", [None, 32,32,3], name="X")
        self.A = tf.placeholder("float", [None, 32,32,3], name="A")
        self.y = tf.placeholder("float", [None, y_dimensions], name="y")
        self.W = []
        self.b = []
        self.L = [self.X]
        for i in range(1, len(self.layer_sizes)):
            self.W.append(tf.Variable(tf.random_normal(
                self.layer_sizes[i-1:i+1], stddev=weight_sd), name='W{}'.format(i)))
            self.b.append(tf.Variable(tf.random_normal(
                [self.layer_sizes[i]], stddev=weight_sd), name='b{}'.format(i)))
        for i, activation in enumerate([nonlinearity for _ in hidden_layers] + [to_logprob]):
            self.L.append(activation(
                tf.add(tf.matmul(self.L[i], self.W[i]), self.b[i])))



        self.out = self.CNN(self.X)

        # Set up optimization
        optimizer = self.optimizer(
            learning_rate=learning_rate, l2_grads=l2_grads, l1_grads=l1_grads, l2_params=l2_params)

        batch_size = min(batch_size, num_examples)
        num_batches = int(np.ceil(num_examples / batch_size))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(num_epochs*num_batches):
                
                percent = i * 20 / (num_epochs*num_batches)                
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" %('=' * int(percent), 5 * percent))
                sys.stdout.flush()

                idx = slice((i % num_batches)*batch_size,((i % num_batches)+1)*batch_size)
                if always_include is not None:
                    A_ = np.vstack((A[always_include], A[idx]))
                    X_ = np.vstack((X[always_include], X[idx]))
                    y_ = np.vstack((y[always_include], y[idx]))

                
                sess.run(optimizer, feed_dict={ self.X: X_, self.y: y_, self.A: A_})
                
            self.W_vals = [weights.eval() for weights in self.W]
            self.b_vals = [biases.eval() for biases in self.b]

    def _initialize_variables(self, sess):
        sess.run(tf.global_variables_initializer())
        for var, val in zip(self.W + self.b, self.W_vals + self.b_vals):
            sess.run(var.assign(val))

    @property
    def log_prob_ys(self):
        return self.L[-1]

    def input_gradients(self, X, y=None, log_scale=True):
        with tf.Session() as session:
            self._initialize_variables(session)
            probs = self.log_prob_ys
            if y is not None:
                probs = probs[:, y]
            if not log_scale:
                probs = tf.exp(probs)
            grads = tf.gradients(probs, self.X)[0].eval(feed_dict={self.X: X})
        return grads

    def largest_gradient_mask(self, X, cutoff=0.67, **kwargs):
        grads = self.input_gradients(X, **kwargs)
        return np.array([np.abs(g) > cutoff*np.abs(g).max() for g in grads])

    def predict_log_proba(self, X):
        with tf.Session() as session:
            self._initialize_variables(session)
            log_probs = self.log_prob_ys.eval(feed_dict={self.X: X})
        return log_probs

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X), axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def grad_explain(self, X, **kwargs):
        yhats = self.predict(X)
        coefs = self.input_gradients(X, **kwargs)
        return [LocalLinearExplanation(X[i], yhats[i], coefs[i]) for i in range(len(X))]

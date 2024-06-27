import tensorflow as tf
import numpy as np
import pickle
from tensorflow.compat.v1 import estimator as tf_estimator

tf.compat.v1.disable_eager_execution()

# Load the data from the generated files
with open("train_images", "rb") as f:
    train_images = pickle.load(f)
with open("train_labels", "rb") as f:
    train_labels = pickle.load(f)
with open("test_images", "rb") as f:
    test_images = pickle.load(f)
with open("test_labels", "rb") as f:
    test_labels = pickle.load(f)

# Model function
def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 1])
    
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=16, kernel_size=[15, 15], padding="valid", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[5, 5], padding="valid", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[5, 5], strides=5)
    
    conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[1, 1], padding="valid", activation=tf.nn.relu)
    
    pool2_flat = tf.reshape(conv3, [-1, 5 * 5 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
    
    logits = tf.layers.dense(inputs=dense, units=10)
    
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    if mode == tf_estimator.ModeKeys.PREDICT:
        return tf_estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    if mode == tf_estimator.ModeKeys.TRAIN:
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.compat.v1.train.get_global_step())
        return tf_estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    eval_metric_ops = {
        "accuracy": tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf_estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Training and evaluation
def main(unused_argv):
    classifier = tf_estimator.Estimator(model_fn=cnn_model_fn, model_dir="tmp/cnn_model3")
    
    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={"x": train_images},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )
    
    logging_hook = tf_estimator.LoggingTensorHook(tensors={"probabilities": "softmax_tensor"}, every_n_iter=50)
    
    classifier.train(input_fn=train_input_fn, hooks=[logging_hook], steps=20000)
    
    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={"x": test_images},
        y=test_labels,
        num_epochs=1,
        shuffle=False
    )
    
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.compat.v1.app.run()

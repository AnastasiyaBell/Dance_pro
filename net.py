import numpy as np
import tensorflow as tf

import dataset


def tf_accuracy(preds, labels):
    return tf.reduce_sum(
        tf.to_float(
            tf.equal(
                tf.argmax(preds, axis=-1, output_type=tf.int32),
                labels
            )
        )
    ) / tf.to_float(tf.shape(labels)[0])


def tf_perplexity(preds):
    log_preds = tf.log(preds)
    inter = tf.exp(tf.reduce_sum((-preds * log_preds), axis=-1))
    return tf.reduce_mean(inter)


def get_inputs_and_labels(iterator, num_dances):
    next_element = iterator.get_next()

    inputs, labels = zip(*next_element)

    inputs = tf.concat(inputs, 0)
    inputs = tf.to_float(tf.reshape(inputs, tf.concat([tf.shape(inputs)[:-1], [3]], 0)))
    labels = tf.concat(labels, 0)
    labels_oh = tf.one_hot(labels, num_dances, dtype=tf.float32)
    return inputs, labels, labels_oh


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


def build_net(x, net_data, stddev, release):
    k_h = 11;
    k_w = 11;
    c_o = 96;
    s_h = 4;
    s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0], trainable=not release)
    conv1b = tf.Variable(net_data["conv1"][1], trainable=not release)
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    k_h = 3;
    k_w = 3;
    s_h = 2;
    s_w = 2;
    padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    k_h = 5;
    k_w = 5;
    c_o = 256;
    s_h = 1;
    s_w = 1;
    group = 2
    conv2W = tf.Variable(net_data["conv2"][0], trainable=not release)
    conv2b = tf.Variable(net_data["conv2"][1], trainable=not release)
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    k_h = 3;
    k_w = 3;
    s_h = 2;
    s_w = 2;
    padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    k_h = 3;
    k_w = 3;
    c_o = 384;
    s_h = 1;
    s_w = 1;
    group = 1
    conv3W = tf.Variable(net_data["conv3"][0], trainable=not release)
    conv3b = tf.Variable(net_data["conv3"][1], trainable=not release)
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    k_h = 3;
    k_w = 3;
    c_o = 384;
    s_h = 1;
    s_w = 1;
    group = 2
    conv4W = tf.Variable(net_data["conv4"][0], trainable=not release)
    conv4b = tf.Variable(net_data["conv4"][1], trainable=not release)
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    k_h = 3;
    k_w = 3;
    c_o = 256;
    s_h = 1;
    s_w = 1;
    group = 2
    conv5W = tf.Variable(net_data["conv5"][0], trainable=not release)
    conv5b = tf.Variable(net_data["conv5"][1], trainable=not release)
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    k_h = 3;
    k_w = 3;
    s_h = 2;
    s_w = 2;
    padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    fc6W = tf.Variable(net_data["fc6"][0], trainable=not release)
    fc6b = tf.Variable(net_data["fc6"][1], trainable=not release)
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

    fc7W = tf.Variable(tf.truncated_normal([4096, 4096], stddev=stddev))
    fc7b = tf.Variable(tf.zeros([4096]))
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

    fc8W = tf.Variable(tf.truncated_normal([4096, 10], stddev=stddev))
    fc8b = tf.Variable(tf.zeros([10]))
    fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

    return fc8


def get_predictions_and_metrics(logits, labels, labels_oh):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_oh))
    preds = tf.nn.softmax(logits)
    accuracy = tf_accuracy(logits, labels)
    perplexity = tf_perplexity(preds)
    return preds, loss, accuracy, perplexity


def get_train_hooks(loss, reg_rate):
    l2_loss = sum(map(tf.nn.l2_loss, tf.get_collection(tf.GraphKeys.WEIGHTS)))
    learning_rate = tf.placeholder(tf.float32)
    opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_op = opt.minimize(loss + reg_rate * l2_loss)
    return train_op, learning_rate


def build_graph(file_names, batch_size, num_dances, reg_rate, stddev, release):
    train_dataset = dataset.build_dataset(file_names['train'], batch_size, num_dances)
    valid_dataset = dataset.build_dataset(file_names['valid'], batch_size, num_dances)
    test_dataset = dataset.build_dataset(file_names['test'], batch_size, num_dances)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)

    training_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(valid_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    inputs, labels, labels_oh = get_inputs_and_labels(iterator, num_dances)

    net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

    logits = build_net(inputs, net_data, stddev, release)

    preds, loss, accuracy, perplexity = get_predictions_and_metrics(logits, labels, labels_oh)

    train_op, learning_rate = get_train_hooks(loss, reg_rate)

    saver = tf.train.Saver(max_to_keep=None)

    hooks = {
        'training_init_op': training_init_op,
        'validation_init_op': validation_init_op,
        'test_init_op': test_init_op,
        'learning_rate': learning_rate,
        'train_op': train_op,
        'saver': saver,
        'loss': loss,
        'accuracy': accuracy,
        'perplexity': perplexity,
        'preds': preds,
    }

    return hooks

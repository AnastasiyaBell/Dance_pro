import os

import tensorflow as tf


LEARNING_RATE_PATIENCE = 10
STOP_PATIENCE = 20
STEP_PERIOD = 100
INIT_LEARNING_RATE = 0.01
DECAY = 0.1

train_results_path = 'results/train'
valid_results_path = 'results/valid'
test_results_path = 'results/test'
checkpoint_path = 'checkpoints'

for p in [train_results_path, valid_results_path, test_results_path, checkpoint_path]:
    if not os.path.isdir(p):
        if os.path.isfile(p):
            os.remove(p)
        os.makedirs(p)


def log(dataset='train', step=None, epoch=None, **kwargs):
    appendix = '.txt' if step is None else '_step.txt'
    first_value = epoch if step is None else step
    for k, v in kwargs.items():
        with open(os.path.join('results', dataset, k + appendix), 'w') as f:
            if dataset == 'test':
                f.write('{}\n'.format(v))
            else:
                f.write('{} {}\n'.format(first_value, v))


def test(dataset):
    init_op = validation_init_op if dataset == 'valid' else test_init_op
    sess.run(init_op)
    count, accumulated_loss, accumulated_acc, accumulated_perpl = 0, 0, 0, 0
    while True:
        try:
            l, acc, perpl = sess.run([loss, accuracy, perplexity], feed_dict={dropout_rate: 0.})
            accumulated_loss += l
            accumulated_acc += acc
            accumulated_perpl += perpl
            count += 1
        except tf.errors.OutOfRangeError:
            break
    accumulated_loss /= count
    accumulated_acc /= count
    accumulated_perpl /= count
    return accumulated_loss, accumulated_acc, accumulated_perpl


step = 0
epoch = 0
lr_impatience = 0
stop_impatience = 0
lr = INIT_LEARNING_RATE

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    l, acc, perpl = test('valid')
    print('EPOCH {} | step {} | loss {:.4} | accuracy {:.4} | perplexity {:.4}'.format(epoch, step, l, acc, perpl))
    log(epoch=epoch, loss=l, accuracy=acc, perplexity=perpl, dataset='valid')
    best_loss = l
    saver.save(os.path.join(checkpoint_path, 'best'))
    while stop_impatience < STOP_PATIENCE:
        sess.run(training_init_op)
        while True:
            try:
                _, l, acc, perpl = sess.run(
                    [train_op, loss, accuracy, perplexity],
                    feed_dict={learning_rate: lr, dropout_rate: 0.5}
                )
                step += 1
                if STEP_PERIOD is not None:
                    if step % STEP_PERIOD == 0:
                        log(step=step, loss=l, accuracy=acc, perplexity=perpl)
                        print('step {} | loss {:.4} | accuracy {:.4}'.format(step, l, acc))
            except tf.errors.OutOfRangeError:
                break
        epoch += 1
        l, acc, perpl = test('valid')
        print('EPOCH {} | step {} | loss {:.4} | accuracy {:.4} | perplexity {:.4}'.format(epoch, step, l, acc, perpl))
        log(epoch=epoch, loss=l, accuracy=acc, perplexity=perpl, dataset='valid')
        if l < best_loss:
            lr_impatience, stop_impatience = 0, 0
            saver.save(os.path.join(checkpoint_path, 'best'))
        else:
            lr_impatience += 1
            stop_impatience += 1
        if lr_impatience >= LEARNING_RATE_PATIENCE:
            lr *= DECAY
    l, acc, perpl = test('test')
    log(loss=l, accuracy=acc, perplexity=perpl, dataset='test')
    print('Testing! EPOCH {} | step {} | loss {:.4} | accuracy {:.4} | perplexity {:.4}'.format(epoch, step, l, acc,
                                                                                                perpl))
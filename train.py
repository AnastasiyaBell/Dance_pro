import os
import socket
import argparse

import tensorflow as tf

import dataset
import net


parser = argparse.ArgumentParser()
parser.add_argument(
    "--release",
    "-r",
    help="If specified all network weights are being trained. "
         "Otherwise only last to layers are training.",
    action="store_true",
)
args = parser.parse_args()


hostname = socket.gethostname()
if hostname == 'aenima':
    DATASET_PATH = os.path.expanduser('~/datasets/letsdance_alexnet')
elif hostname == 'lateralus':
    DATASET_PATH = os.path.expanduser('~/datasets/letsdance_alexnet')
else:
    raise ValueError("The script is run on unknown device. "
                     "Can not find dataset. Please specify dataset "
                     "path for your device in script "
                     "{} in variable PATH".format(__file__))

TRAIN_PATH = "letsdance_split/train"
VALID_PATH = "letsdance_split/validation"
TEST_PATH = "letsdance_split/test"

LEARNING_RATE_PATIENCE = 10
STOP_PATIENCE = 20
STEP_PERIOD = 100
INIT_LEARNING_RATE = 0.01
DECAY = 0.1

REG_RATE = 5e-4
STDDEV = 0.005

file_names = dataset.select_train_valid_test_file_names(
    DATASET_PATH,
    TRAIN_PATH,
    VALID_PATH,
    TEST_PATH,
)

BATCH_SIZE = 30
NUM_DANCES = len(file_names['train'])

hooks = net.build_graph(file_names, BATCH_SIZE, NUM_DANCES, REG_RATE, STDDEV, args.release)

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


def test(dataset, hooks):
    init_op = hooks['validation_init_op'] if dataset == 'valid' else hooks['test_init_op']
    sess.run(init_op)
    count, accumulated_loss, accumulated_acc, accumulated_perpl = 0, 0, 0, 0
    while True:
        try:
            l, acc, perpl = sess.run([hooks['loss'], hooks['accuracy'], hooks['perplexity']])
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
    l, acc, perpl = test('valid', hooks)
    print('EPOCH {} | step {} | loss {:.4} | accuracy {:.4} | perplexity {:.4}'.format(epoch, step, l, acc, perpl))
    log(epoch=epoch, loss=l, accuracy=acc, perplexity=perpl, dataset='valid')
    best_loss = l
    hooks['saver'].save(sess, os.path.join(checkpoint_path, 'best'))
    while stop_impatience < STOP_PATIENCE:
        sess.run(sess, hooks['training_init_op'])
        while True:
            try:
                _, l, acc, perpl = sess.run(
                    [hooks['train_op'], hooks['loss'], hooks['accuracy'], hooks['perplexity']],
                    feed_dict={hooks['learning_rate']: lr}
                )
                step += 1
                if STEP_PERIOD is not None:
                    if step % STEP_PERIOD == 0:
                        log(step=step, loss=l, accuracy=acc, perplexity=perpl)
                        print('step {} | loss {:.4} | accuracy {:.4}'.format(step, l, acc))
            except tf.errors.OutOfRangeError:
                break
        epoch += 1
        l, acc, perpl = test('valid', hooks)
        print('EPOCH {} | step {} | loss {:.4} | accuracy {:.4} | perplexity {:.4}'.format(epoch, step, l, acc, perpl))
        log(epoch=epoch, loss=l, accuracy=acc, perplexity=perpl, dataset='valid')
        if l < best_loss:
            lr_impatience, stop_impatience = 0, 0
            hooks['saver'].save(sess, os.path.join(checkpoint_path, 'best'))
        else:
            lr_impatience += 1
            stop_impatience += 1
        if lr_impatience >= LEARNING_RATE_PATIENCE:
            lr *= DECAY
    l, acc, perpl = test('test')
    log(loss=l, accuracy=acc, perplexity=perpl, dataset='test')
    print(
        'Testing! EPOCH {} | step {} | loss {:.4} | accuracy {:.4} | perplexity {:.4}'.format(
            epoch, step, l, acc, perpl
        )
    )

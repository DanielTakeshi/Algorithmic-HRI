"""
OK, using what I've learned, I **should** be able to build a network that can
predict actions easily. This is NOT a "Q-Network" in the sense that we don't use
target values. However it DOES map from 'phi's (i.e. 4 consecutive game frames)
to predicted actions. Also, the actions are going to be limited to a subset of
the popular ones. This means we still probably need some randomness in our
actions if we want the agent to FIRE in Breakout, for instance.

What's the goal? I want to run this file on data that I created from my own
gameplay. Then I will save the weights in some file. Then in my deep_q_rl fork
of spragnur's code, I will add an extra class which takes the weights as input
and will get called upon whenever we have to decide on random actions.

TODO LONG TERM: may want to figure out ways I can add more parameters, etc.,
better get a full pipeline for coding down. But I think that's longer-term.

(c) December 2016 by Daniel Seita, heavily based off of spragnur's code and the
Lasagne tutorial.
"""

from __future__ import print_function
import os
import sys
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import l2, l1

def load_datasets(path):
    """ 
    Loads the data. It requires the following assumptions:
    - That the data is small enough to fit in RAM. For now this should be met.
    - That actions are re-aligned to [0,1,2,...] instead of non-consecutively.
    - That they are stored as (N, depth, width, height), with depth=4 usually.
    """
    X_train = np.load(path+ "train.data.npy").astype('float32')
    y_train = np.load(path+ "train.labels.npy").astype('int32')
    X_val = np.load(path+ "valid.data.npy").astype('float32')
    y_val = np.load(path+ "valid.labels.npy").astype('int32')
    X_test = np.load(path+ "test.data.npy").astype('float32')
    y_test = np.load(path+ "test.labels.npy").astype('int32')
    print("All the data has been loaded. Here are the dimensions:")
    print("  X_train.shape={}, y_train.shape={}".format(X_train.shape,y_train.shape))
    print("  X_val.shape={}, y_val.shape={}".format(X_val.shape,y_val.shape))
    print("  X_test.shape={}, y_test.shape={}\n".format(X_test.shape,y_test.shape))
    return X_train, y_train, X_val, y_val, X_test, y_test


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """ 
    See the Lasagne documentation for details on this. It doesn't return the
    last minibatch if it isn't a multiple of the batchsize, but that's OK.
    """
    assert len(inputs) == len(targets) 
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def build_nips_network_dnn(input_var, input_width, input_height, output_dim,
                           num_frames, batch_size):
    """
    Build a network consistent with the 2013 NIPS paper. Daniel: note that the
    method from He didn't get published until 2015 which is why the NIPS paper
    doesn't have it. Also, we should consider using the softmax.
    """
    from lasagne.layers import dnn

    l_in = lasagne.layers.InputLayer(
        shape=(None, num_frames, input_width, input_height),
        input_var=input_var
    )

    l_conv1 = dnn.Conv2DDNNLayer(
        l_in,
        num_filters=16,
        filter_size=(8, 8),
        stride=(4, 4),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        #W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )

    l_conv2 = dnn.Conv2DDNNLayer(
        l_conv1,
        num_filters=32,
        filter_size=(4, 4),
        stride=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        #W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )

    l_hidden1 = lasagne.layers.DenseLayer(
        l_conv2,
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        #W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )

    # Daniel: no softmax used here?!?
    l_out = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=output_dim, # Daniel: i.e. number of actions.
        nonlinearity=lasagne.nonlinearities.softmax, # Daniel: spragnur had this as None
        W=lasagne.init.HeUniform(),
        #W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )
    return l_out



def main(reg_type='l1', reg=0.0, num_epochs=100, batch_size=32, batches_per_epoch=None):
    """ Runs the whole pipeline. """
    path = "/home/daniel/Algorithmic-HRI/final_data/breakout/"
    X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(path=path)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Build the network. Unlike spragnur's code, I will need the input variables
    # because I don't want to bother using shared variables (for now).
    network = build_nips_network_dnn(input_var=input_var,
                                     input_width=84, 
                                     input_height=84, 
                                     output_dim=3, # Breakout num actions
                                     num_frames=4, 
                                     batch_size=batch_size)
    print("Finished builing the network.")
    params = lasagne.layers.get_all_params(network, trainable=True)

    # Create a loss expression for training, i.e., a scalar objective we want to
    # minimize (for our multi-class problem, it is the cross-entropy loss):
    # Then add regularize_network_params (NOT regularize_layer_params!!).
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction,target_var)
    loss = loss.mean()
    if (reg_type == 'l1'):
        loss = loss + reg*lasagne.regularization.regularize_network_params(network,l1)
    elif (reg_type == 'l2'):
        loss = loss + reg*lasagne.regularization.regularize_network_params(network,l2)
    else:
        raise ValueError("reg_type={} is not suported".format(reg_type))

    # Create update expressions for training, i.e., how to modify the parameters
    # at each training step. Daniel: I'm using rmsprop.
    lr = 0.0002
    rho = 0.99
    rms_epsilon = 1e-6
    updates = lasagne.updates.rmsprop(loss, params, lr, rho, rms_epsilon)

    # Create a loss expression for validation/testing. (Daniel: I borrowed this
    # from the tutorial, but since we don't use dropout, might not be needed.)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
    test_loss = test_loss.mean()
    if (reg_type == 'l1'):
        test_loss = test_loss + reg*lasagne.regularization.regularize_network_params(network,l1)
    elif (reg_type == 'l2'):
        test_loss = test_loss + reg*lasagne.regularization.regularize_network_params(network,l2)
    else:
        raise ValueError("reg_type={} is not suported".format(reg_type))
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function for validation (with no parameter updates).
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # In each epoch, we do a full pass over the training data.
    # Daniel: can also change this to see validation performance more often.
    print("Starting training...")

    for epoch in range(num_epochs):
        # This will do the training, with train_fn called (with weight updates).
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
            if batches_per_epoch != None and train_batches == batches_per_epoch:
                break

        # Training accuracy (I'm just curious). Must use val_fn so weights are same.
        train2_err = 0
        train2_acc = 0
        train2_batches = 0
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            train2_err += err
            train2_acc += acc
            train2_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch only (not a moving average):
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t\t{:.2f} %".format(train2_acc / train2_batches * 100))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err/test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc/test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == "__main__":
    main(reg_type='l1', reg=0.001, num_epochs=20, batch_size=32, batches_per_epoch=None)

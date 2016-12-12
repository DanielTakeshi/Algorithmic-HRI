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

Note: In spragnur's DQN code, load network weights like this:

    with np.load('model.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

Updates November 28, 2016: I've added a lot of functionality to analyze the
classifier.

(c) November 2016 by Daniel Seita, heavily based off of spragnur's code and the
Lasagne tutorial.
"""

from __future__ import print_function
import os
import sys
import time
import numpy as np
np.set_printoptions(suppress=True)
import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import l2, l1
import utilities
from PIL import Image
import cv2


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


def build_nature_network_dnn(input_width, input_height, output_dim, num_frames,
                             batch_size, human_net, input_var):
    """
    Build a large network consistent with the DeepMind Nature paper.
    """
    from lasagne.layers import dnn

    if human_net:
        l_in = lasagne.layers.InputLayer(
            shape=(None, num_frames, input_width, input_height),
            input_var=input_var
        )
    else:
        l_in = lasagne.layers.InputLayer(
            shape=(None, num_frames, input_width, input_height)
        )

    l_conv1 = dnn.Conv2DDNNLayer(
        l_in,
        num_filters=32,
        filter_size=(8, 8),
        stride=(4, 4),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_conv2 = dnn.Conv2DDNNLayer(
        l_conv1,
        num_filters=64,
        filter_size=(4, 4),
        stride=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_conv3 = dnn.Conv2DDNNLayer(
        l_conv2,
        num_filters=64,
        filter_size=(3, 3),
        stride=(1, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_hidden1 = lasagne.layers.DenseLayer(
        l_conv3,
        num_units=512,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_out = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax, # Daniel: new
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    return l_out


# Daniel: TODO change this to what I actually have in deep_q_rl/make_net.py
# Oh, with the exception of the softmax, of course.
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
        #W=lasagne.init.HeUniform(),
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )

    l_conv2 = dnn.Conv2DDNNLayer(
        l_conv1,
        num_filters=32,
        filter_size=(4, 4),
        stride=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify,
        #W=lasagne.init.HeUniform(),
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )

    l_hidden1 = lasagne.layers.DenseLayer(
        l_conv2,
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify,
        #W=lasagne.init.HeUniform(),
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )

    # Daniel: no softmax used here?!?
    l_out = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=output_dim, # Daniel: i.e. number of actions.
        nonlinearity=lasagne.nonlinearities.softmax, # Daniel: spragnur had this as None
        #W=lasagne.init.HeUniform(),
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(.1)
    )
    return l_out



def do_training(X_train, y_train, X_val, y_val, X_test, y_test, reg_type='l1',
                reg=0, num_epochs=100, batch_size=32, out_dir='qnet_out',
                num_acts=3):
    """ 
    Runs the whole pipeline. Can call this multiple times during process of
    hyperparameter tuning.
    """
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Build the network. Unlike spragnur's code, I will need the input variables
    # because I don't want to bother using shared variables (for now).
    network = build_nature_network_dnn(input_width=84, 
                                       input_height=84,
                                       output_dim=num_acts,
                                       num_frames=4,
                                       batch_size=batch_size,
                                       human_net=True, 
                                       input_var=input_var)
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
    # Also compile a second function for validation (with no parameter updates).
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # In each epoch, we do a full pass over the training data.
    # Daniel: can also change this to see validation performance more often.
    print("Starting training...")
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    train_times = []

    for epoch in range(num_epochs):
        # This will do the training, with train_fn called (with weight updates).
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
        train_time = time.time() - start_time

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
        # The train_loss is the average train loss over all minibatches, etc.
        train_loss = train_err / train_batches
        train_acc = train2_acc / train2_batches * 100
        valid_loss = val_err / val_batches
        valid_acc = val_acc / val_batches * 100
        print("Epoch {} of {}, training took {:.3f}s".format(epoch+1,num_epochs,train_time))
        print("  training loss:\t\t{:.6f}".format(train_loss))
        print("  training accuracy:\t\t{:.2f} %".format(train_acc))
        print("  validation loss:\t\t{:.6f}".format(valid_loss))
        print("  validation accuracy:\t\t{:.2f} %".format(valid_acc))
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        train_times.append(train_time)

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
    test_loss = test_err/test_batches
    test_acc = test_acc/test_batches * 100
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_loss))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc))
    
    # Save statistics and dump the network weights to (two) files, w/keywords.
    stem = reg_type+ '_' +str(reg)+ '_epochs_' +str(num_epochs)+ '_bsize_' \
           +str(batch_size)
    np.savez(out_dir+'stats_'+stem, train_losses=train_losses,
                                    train_accs=train_accs, 
                                    valid_losses=valid_losses,
                                    valid_accs=valid_accs,
                                    train_times=train_times,
                                    test_loss_acc=[test_loss,test_acc])
    np.savez(out_dir+'model_'+stem, *lasagne.layers.get_all_param_values(network))


def train():
    path = "/home/daniel/Algorithmic-HRI/final_data/breakout/"
    na = 3 # Change according to game!
    out = 'qnet_outputs/qnet_breakout_nature/' # change!
    utilities.make_path_check_exists(out)
    regs = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(path=path)

    for r in regs:
        print("\nCurrently on regularization r={}".format(r))
        print("L1 regularization.\n")
        do_training(X_train, y_train, X_val, y_val, X_test, y_test, reg_type='l1',
                reg=r, num_epochs=30, batch_size=32, out_dir=out, num_acts=na)
        print("\n Now L2 regularization.\n")
        do_training(X_train, y_train, X_val, y_val, X_test, y_test, reg_type='l2',
                reg=r, num_epochs=30, batch_size=32, out_dir=out, num_acts=na)
    print("\nWhew! All done with everything.")


def sandbox_test():
    """ Sandbox testing, so I can analyze output, provide examples in a paper,
    etc. Needs to build the same network and load weights. If I'm using the
    November 26-th version, this should result in 86.41% accuracy on the full
    testing data.
    """

    # Check ALL of these:
    path = "final_data/space_invaders/"
    #model_file = "qnet_outputs/qnet_breakout_nature/model_l2_0.01_epochs_30_bsize_32.npz"
    model_file = "qnet_outputs/qnet_spaceinv_nature/model_l1_0.0005_epochs_30_bsize_32.npz"
    out = "tmp/"
    batch_size = 32
    num_acts = 6 # breakout=3, SI=6

    X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(path=path)
    utilities.make_path_check_exists(out)

    # Now try to replicate the training method in re-generating the network.
    # TODO may want to make this easier to duplicate from the train() method?
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    network = build_nature_network_dnn(input_width=84, 
                                       input_height=84,
                                       output_dim=num_acts,
                                       num_frames=4,
                                       batch_size=batch_size,
                                       human_net=True, 
                                       input_var=input_var)
    print("Finished builing the network.")
    with np.load(model_file) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    # The 'network' has weights loaded, so we can get last layer and predict.
    # The final layer contains the softmax values, so take the arg-max.
    final_layer = lasagne.layers.get_output(network, deterministic=True)
    acc = T.mean(T.eq(T.argmax(final_layer, axis=1), target_var),
                      dtype=theano.config.floatX)
    val_fn = theano.function([input_var, target_var], [acc, final_layer])

    test_acc = 0
    test_batches = 0
    all_a_probs = np.zeros((X_test.shape[0]-(X_test.shape[0]%batch_size), num_acts))
    print("Now testing (and saving probabilities in all_a_probs, "+
          "shape={}".format(all_a_probs.shape))

    for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
        inputs, targets = batch
        acc, a_probs = val_fn(inputs, targets)
        all_a_probs[batch_size*test_batches:batch_size*(test_batches+1)] = a_probs
        test_acc += acc
        test_batches += 1
    test_acc = test_acc/test_batches * 100
    print("Final results:")
    print("  test accuracy:\t\t{:.2f} %".format(test_acc))
    np.savetxt("test_action_preds.txt", all_a_probs, fmt='%1.5f')


def save_phi(phi, game, index, padding):
    """ This will save 'phi' to output as a series of stitched images. Heh, who
    would have thought that I'd use code from a blog post for research the
    following day?
    """
    blank_image = Image.new("RGB", (84*4+30, 84), "white")
    coords = [(0,0), (84+10,0), (84*2+20,0), (84*3+30,0)]
    for i in range(phi.shape[0]):
        f = Image.fromarray(phi[i])
        blank_image.paste(f, coords[i])
    name = str(index).zfill(padding)
    blank_image.save("qnet_output/" +game+ "_phi_" +name+ ".png")


def do_analysis_testing_v1(i=0):
    """ 
    With the action output file, let's inspect the predictions.  I manually
    created a file with (predicted probs)-(target) to make inspection easy.
    Don't forget, vim numbers lines starting from 1, but it's 0-indexed!! This
    method lets me visually inspect an index and save its image via save_phi.
    """
    #path = "final_data/breakout/"
    path = "final_data/space_invaders/"
    #aprobs = np.loadtxt("qnet_outputs/qnet_breakout_nature/test_action_preds_breakout_nature.txt")
    aprobs = np.loadtxt("qnet_outputs/qnet_spaceinv_nature/test_action_preds_spaceinv_nature.txt")
    X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(path=path)
    X_test = X_test[:13248]  # 5024 for Breakout
    y_test = y_test[:13248]  # 5024 for Breakout
    print("Loaded data. X_test.shape = {}\ny_test.shape = {}\naprobs.shape = {}.".format(
        X_test.shape, y_test.shape, aprobs.shape))
    print("First few y_tests:\n{}".format(y_test[:20]))
    print("First few aprobs:\n{}".format(aprobs[:20]))
    np.savetxt("test_action_targets.txt", y_test.reshape((13248,1)), fmt='%d') # do this once

    # This will save the phi-s that I want.
    #save_phi(phi=X_test[i], game='breakout', index=i, padding=4)


def do_analysis_testing_v2():
    """ 
    This will use the qnet_output/test_action_pairs.txt file to do more rigorous
    analysis, e.g. which classes did it mess up on, worst case scenarios, etc.
    Note the 'pairs' file have the true label, i.e. I have to do that by
    explicitly going into vim, etc., and it starts by calling from the previous
    analysis testing v1 method.
    """
    #num_acts = 3
    num_acts = 6
    #path = "final_data/breakout/"
    path = "final_data/space_invaders/"
    #aprob_pairs = np.loadtxt("qnet_outputs/qnet_breakout_nature/test_action_pairs_breakout_nature.txt")
    aprob_pairs = np.loadtxt("qnet_outputs/qnet_spaceinv_nature/test_action_pairs_spaceinv_nature.txt")

    print("aprob_pairs.shape={}".format(aprob_pairs.shape))
    print("\nFirst few action probabilities, followed by arg-max for them.")
    print((aprob_pairs[:,:num_acts])[:20])
    print((np.argmax(aprob_pairs[:,:num_acts], axis=1))[:20])

    # Separate data into indices for correct (and incorrect) predictions.
    inds_correct = np.where(np.argmax(aprob_pairs[:,:num_acts], axis=1) == aprob_pairs[:,num_acts])[0]
    inds_wrong = np.where(np.argmax(aprob_pairs[:,:num_acts], axis=1) != aprob_pairs[:,num_acts])[0]

    # Get per-class accuracy
    correct = np.zeros(num_acts)
    total = np.zeros(num_acts)
    for (index,row) in enumerate(aprob_pairs):
        target = int(row[num_acts])
        if np.argmax(row[:num_acts]) == target:
            correct[target] += 1
        total[target] += 1
    print(correct)
    print(total)
    print(correct.astype('float')/total)


if __name__ == "__main__":
    """
    Choose which of the methods I want to do. Generally I won't be running all
    of these at once. Apologies, there's a LOT of assumptions here. The code
    isn't quite general enough yet. =(
    """
    #train()
    #sandbox_test()
    #do_analysis_testing_v1(i=653)
    #do_analysis_testing_v2()
    print("")

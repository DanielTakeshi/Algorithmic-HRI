"""
Test theano to see if I can get stuff working.
Based on code from Nathan Sprague's project.
Modified by Daniel Seita
"""

import numpy as np
import lasagne
import numpy as np
import theano
import theano.tensor as T
import sys


class DeepQLearner:
    """ Deep Q-learning network using Lasagne. """

    def __init__(self, input_width, input_height, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, input_scale=255.0):
        """
        I think input_scale is correct. My data is in the raw phi stuff, I never
        changed that. Be careful, there are several locations where I changed
        this to have 4 frames instead of 5.
        """
        self.input_width = input_width
        self.input_height = input_height
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.discount = discount
        self.rho = rho
        self.lr = learning_rate
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self.rng = rng

        # Daniel: for Lasagne, we only need to pass the *output* layer.
        lasagne.random.set_rng(self.rng)
        self.update_counter = 0
        self.l_out = self.build_network(network_type, input_width, input_height,
                                        num_actions, num_frames, batch_size)
        if self.freeze_interval > 0:
            self.next_l_out = self.build_network(network_type, input_width,
                                                 input_height, num_actions,
                                                 num_frames, batch_size)
            self.reset_q_hat()

        states = T.tensor4('states')
        next_states = T.tensor4('next_states')
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')

        # Shared variables for training from a minibatch of replayed
        # state transitions, each consisting of num_frames + 1 (due to
        # overlap) images, along with the chosen action and resulting
        # reward and terminal status. (Daniel: so wait, I can't just put in
        # my PHIs? I know it's overlaping images but with batch size 32, that
        # means we have 32 phi's, right?)
        self.imgs_shared = theano.shared(
            # Daniel: let me try with just num_frames here ...
            np.zeros((batch_size, num_frames, input_height, input_width),
                    dtype=theano.config.floatX))
            # np.zeros((batch_size, num_frames + 1, input_height, input_width),
            #         dtype=theano.config.floatX))
        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))
        self.actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))
        self.terminals_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        # Shared variable for a single state, to calculate q_vals.
        # Daniel: yes, this is good, it's ONE phi here.
        self.state_shared = theano.shared(
            np.zeros((num_frames, input_height, input_width),
                     dtype=theano.config.floatX))

        q_vals = lasagne.layers.get_output(self.l_out, states / input_scale)
        
        if self.freeze_interval > 0:
            next_q_vals = lasagne.layers.get_output(self.next_l_out,
                                                    next_states / input_scale)
        else:
            next_q_vals = lasagne.layers.get_output(self.l_out,
                                                    next_states / input_scale)
            next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

        terminalsX = terminals.astype(theano.config.floatX)
        actionmask = T.eq(T.arange(num_actions).reshape((1, -1)),
                          actions.reshape((-1, 1))).astype(theano.config.floatX)

        target = (rewards +
                  (T.ones_like(terminalsX) - terminalsX) *
                  self.discount * T.max(next_q_vals, axis=1, keepdims=True))
        output = (q_vals * actionmask).sum(axis=1).reshape((-1, 1))
        diff = target - output

        # Daniel: huh, are we really taking squared difference as the loss? I
        # thought the cross entropy loss would be more appropriate.
        if self.clip_delta > 0:
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            # 
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = T.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        if batch_accumulator == 'sum':
            loss = T.sum(loss)
        elif batch_accumulator == 'mean':
            loss = T.mean(loss)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))

        # Daniel: I think this is similar to the CS 294-129 homework where we
        # can put all parameters in a dictionary. Convenient helper function!
        params = lasagne.layers.helper.get_all_params(self.l_out)  
        train_givens = {
            states: self.imgs_shared,
            #states: self.imgs_shared[:, :-1], # Daniel: try this
            #next_states: self.imgs_shared[:, 1:], # Daniel: try this
            next_states: self.imgs_shared,
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }
        # Daniel: I got rid of deep mind's rule.
        if update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho,
                                              self.rms_epsilon)
        elif update_rule == 'sgd':
            updates = lasagne.updates.sgd(loss, params, self.lr)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))

        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)

        # Daniel: This is a standard Theano function with nothing as input, loss
        # as output, updates specified by the lasagne momentum variable, and the
        # 'given' variables in Theano (I think this means the variables are part
        # of the computation, but we don't want to take them as input).
        self._train = theano.function([], [loss], updates=updates,
                                      givens=train_givens)

        q_givens = {
            states: self.state_shared.reshape((1,
                                               self.num_frames,
                                               self.input_height,
                                               self.input_width))
        }
        self._q_vals = theano.function([], q_vals[0], givens=q_givens)


    def train(self, imgs, actions, rewards, terminals):
        """
        Train one batch. (Daniel: from spragnur's code, he calls this only (I
        think) in the ale_agent class, and then he uses the dataset to extract
        the necessary input.)

        Arguments:

        imgs - b x (f + 1) x h x w numpy array, where b is batch size,
               f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """
        self.imgs_shared.set_value(imgs)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)
        if (self.freeze_interval > 0 and
            self.update_counter % self.freeze_interval == 0):
            self.reset_q_hat()
        loss = self._train()
        self.update_counter += 1
        return np.sqrt(loss)


    def q_vals(self, state):
        """  TODO document  """
        self.state_shared.set_value(state)
        return self._q_vals()


    def choose_action(self, state, epsilon):
        """ 
        Daniel: I can probably get some classification stuff here. 
        The self._train method returns a **loss** which is less intuitive.
        """
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.q_vals(state)
        return np.argmax(q_vals)


    def reset_q_hat(self):
        """  TODO document  """
        all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        lasagne.layers.helper.set_all_param_values(self.next_l_out, all_params)


    def build_network(self, network_type, input_width, input_height,
                      output_dim, num_frames, batch_size):
        """
        TODO Document this. ONLY USING NIPS FOR NOW ...
        """
        return self.build_nips_network_dnn(input_width, input_height,
                                           output_dim, num_frames, batch_size)


    def build_nips_network_dnn(self, input_width, input_height, output_dim,
                               num_frames, batch_size):
        """
        Build a network consistent with the 2013 NIPS paper.  Daniel: this is
        what's interesting. The input is num_frames. So why are we having
        num_frames + 1 in other parts of this code?? Has to do with Q-Learning
        and targets, but here I don't want to worry about that! PS: This is a
        lot clearer after reading Theano and Lasagne documentation. Also,
        really, they used the Gaussian initialization instead of He-style? [OH,
        He wasn't introduced until 2014 ... never mind.] Finally, I think there
        should be a softmax for the final layer.
        """
        from lasagne.layers import dnn

        l_in = lasagne.layers.InputLayer(
            shape=(None, num_frames, input_width, input_height)
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
            nonlinearity=None,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        return l_out


if __name__ == "__main__":
    path = "/Users/danielseita/Algorithmic-HRI/final_data/breakout/"
    train_data = np.load(path+ "train.data.npy").astype('float32')
    train_labels = np.load(path+ "train.labels.npy").astype('int32')
    valid_data = np.load(path+ "valid.data.npy").astype('float32')
    valid_labels = np.load(path+ "valid.labels.npy").astype('int32')
    test_data = np.load(path+ "test.data.npy")
    test_labels = np.load(path+ "test.labels.npy")
    print("All the data has been loaded.")

    # Daniel: might want to adjust some of these values ...
    net = DeepQLearner(input_width=84, 
                       input_height=84, 
                       num_actions=3, # Daniel: this WAS 16, I change to 3.
                       num_frames=4,  # Daniel: this is correct, phi_length.
                       discount=.95, 
                       learning_rate=.0002, 
                       rho=.99, 
                       rms_epsilon=1e-6, 
                       momentum=0,
                       clip_delta=0,
                       freeze_interval=-1,
                       batch_size=32,
                       network_type='nips_dnn',
                       update_rule='rmsprop',
                       batch_accumulator='mean', 
                       rng=np.random.RandomState())

    # Daniel: let's try training it using all the data I have.
    print("Built the network. Let's train it.")

    max_iter = 1000
    batch_size = 32
    N = len(train_labels) 
    rewards = np.ones((batch_size,1)).astype('float32')
    terminals = np.zeros((batch_size,1), dtype=bool)

    # Daniel: don't forget this!!
    valid_actions = np.zeros((len(valid_labels),1)).astype('int32')
    for (j,val) in enumerate(valid_labels):
        if val == 3:
            valid_actions[j,0] = 1
        elif val == 4:
            valid_actions[j,0] = 2

    # Daniel: OK let's try getting this to work!!
    for i in range(max_iter):
        indices = np.random.choice(N, size=batch_size, replace=False)
        data = train_data[indices].transpose(0,3,1,2)
        labels = train_labels[indices].reshape((batch_size,1))

        # Daniel: not sure, we may have to scale into 0,1,2 ...
        # AH, that changes the loss!!!
        actions = np.zeros((batch_size,1)).astype('int32')
        for (j,val) in enumerate(labels):
            if val == 3:
                actions[j,0] = 1
            elif val == 4:
                actions[j,0] = 2

        loss = net.train(data, actions, rewards, terminals)
        if i % 10 == 0:
            print("i={}, loss={}".format(i,loss))

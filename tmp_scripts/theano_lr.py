# Based on documentation so I understand what's going on.
# Daniel Seita

import numpy as np
import theano
import theano.tensor as T 
rng = np.random

N = 500
feats = 784

# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 1000

# Declare Theano symbolic variables (Daniel: the variable names are optional).
x = T.dmatrix("x")
y = T.dvector("y")

# initialize the weight vector w randomly
#
# this and the following bias variable b are shared so they keep their values
# between training iterations (updates) (Daniel: yes! I was right on this).
# Daniel: w and b are shared across multiple functions, and they are actually
# functions themselves ("shared" is a theano function).
w = theano.shared(rng.randn(feats), name="w") # initialize the bias term
b = theano.shared(0., name="b")

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))
prediction = p_1 > 0.5
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)   # Cross-entropy loss function 
cost = xent.mean() + 0.01 * (w ** 2).sum()      # The cost to minimize (w/regularization!)
gw, gb = T.grad(cost, [w, b])                   # Compute gradient of cost wrt w and b.

# Now we compile it. Daniel: prediction,xent as output makes sense, that's the
# stuff we want to see (and it means we can print out predictions). The updates
# are the gradients! Very nice ... syntax is (original_variable, modifications).
# So I guess we will need to repeatedly call 'train', in a for loop.
train = theano.function(
    inputs=[x, y],
    outputs=[prediction, xent],
    updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb))
)

# Daniel: Let's do prediction. Note that there's nothing to update.
predict = theano.function(inputs=[x], outputs=[prediction])

# Daniel: Now train. The theano.function handles the gradient updates. Nice!
# If we were using neural networks, we'd probably want x and y to represent
# minibatches instead of the full data, but here it's fine. Don't forget, we're
# not actually putting in x and y, we're really using D[0] and D[1] because
# that's the raw data. The x and y are like parameters! That's the tricky thing,
# I have to be careful to view x's, y's, etc as parameters which can take on
# different values each time, not like simple variable assignment!
for i in range(training_steps):
    pred,err = train(D[0],D[1])
    if i % 10 == 0:
        num_correct = float(np.sum(predict(D[0])))
        print("Iteration i={}, accuracy={:.4f}".format(i, num_correct/N))

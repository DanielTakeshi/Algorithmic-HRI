This directory will contain data files relating to tuning neural network
hyperparameters on my gameplay data. I will also have information about the
predictions, and examples of the 'phi's which were easiest/hardest to classify.
Right now it contains all the model files tested, along with the statistics
files.

Note: I am keeping most of the data private for now since it would be a lot of
MBs to add to GitHub.

Breakout NIPS:

Breakout Nature:

OK this one works now. Best is 0.01 L2.

Space Invaders Nature:

The best performing file is model_l1_0.0005_epochs_30_bsize_32.npz which I
subsequently copied over to the deep_q_rl directory.

EDIT: Argh, i may have to re-run SI since I overwrite some of it with Breakout
data. But the best performing model file should still be OK and that's all that
matters.

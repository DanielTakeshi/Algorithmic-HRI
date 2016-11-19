# Algorithmic-HRI
My final project for CS 294-115, then hopefully I can build on it for later.

Steps:

(1) Play games and gather data, using [TODO describe]

(2) Now run the preprocessing script to generate the dataset, to be used for
training classifier from (sequence of frames) -> one action (for the
action taken based on the last frame in the sequence of frames), OR we can store
those as datapoints.

(3) Also don't forget checkpoint replay.

(4) Then run training scripts.

TODO clean this up ...

After running scripts/process_data, get something lke:

$ls -lh final_data/breakout/
total 7.1G
-rw-rw-r-- 1 daniel daniel 1.5G Nov 19 13:15 test.data.npy
-rw-rw-r-- 1 daniel daniel  53K Nov 19 13:15 test.labels.npy
-rw-rw-r-- 1 daniel daniel 5.4G Nov 19 13:15 train.data.npy
-rw-rw-r-- 1 daniel daniel 200K Nov 19 13:15 train.labels.npy
-rw-rw-r-- 1 daniel daniel 289M Nov 19 13:15 valid.data.npy
-rw-rw-r-- 1 daniel daniel  11K Nov 19 13:15 valid.labels.npy

And so these have shape (for instance) (N,84,84,4). Then I can use some of the
MNIST data examples online to transform those into TFRecords files!

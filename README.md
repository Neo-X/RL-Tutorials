# RL-Tutorials

This work is done to help understand the full Reiforcement Learning pipeline. I found a lack of simple, easy to understand examples online to illustrate the RL process and how Neural Networks are used as a good function aproximator.

## Dependancies

 1. sudo apt-get -y install liblapack3 liblapack-dev libblas3 libblas-dev gfortran libspqr1.3.1 libcholmod2.1.2 libmetis5 libmetis-dev libcolamd2.8.0 libccolamd2.8.0 libcamd2.3.1 libamd2.3.1 libx11-dev python-dev
 2. pip install Theano
 3. pip install matplotlib
 4. pip install Lasagne==0.1
 5. sudo apt-get install python-pyode
 6. sudo apt-get install python-opengl

### To record videos

You need ffmpeg
  1. sudo add-apt-repository ppa:mc3man/trusty-media
  2. sudo apt-get update
  3. sudo apt-get install ffmpeg gstreamer0.10-ffmpeg

## Using

	$ python RunGame.py Deep.json


## References

 1. https://github.com/Newmu/Theano-Tutorials
 2. https://github.com/spragunr/deep_q_rl


## Games
 The different example games.

### BallGame 1D

python RunBallGame1D.py settings/BallGame1D/DeepCACLA.json

# NeRF uncertainty with Ensemble approach

This implementation is built upon PyTorch re-implementation of [nerf-pytorch](https://github.com/krrish94/nerf-pytorch).

## What is a NeRF?

### [Project](http://tancik.com/nerf) | [Video](https://youtu.be/JuH79E8rdKc) | [Paper](https://arxiv.org/abs/2003.08934)

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://tancik.com/nerf)  
 [Ben Mildenhall](https://people.eecs.berkeley.edu/~bmild/)\*<sup>1</sup>,
 [Pratul P. Srinivasan](https://people.eecs.berkeley.edu/~pratul/)\*<sup>1</sup>,
 [Matthew Tancik](http://tancik.com/)\*<sup>1</sup>,
 [Jonathan T. Barron](http://jonbarron.info/)<sup>2</sup>,
 [Ravi Ramamoorthi](http://cseweb.ucsd.edu/~ravir/)<sup>3</sup>,
 [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html)<sup>1</sup> <br>
 <sup>1</sup>UC Berkeley, <sup>2</sup>Google Research, <sup>3</sup>UC San Diego  
  \*denotes equal contribution

A neural radiance field is a simple fully connected network (weights are ~5MB) trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views.

Optimizing a NeRF takes between a few hours and a day or two (depending on resolution) and only requires a single GPU. Rendering an image from an optimized NeRF takes somewhere between less than a second and ~30 seconds, again depending on resolution.

## NeRF Ensemble

<!-- TO BE WRITTEN -->

<!-- Research papers source -->

![NeRF Uncertainty Ensemble pipeline](https://github.com/CTW121/NeRF-Uncertainty-Ensemble/blob/master/images/Ensemble_pipeline.png)

<!-- EXPLAIN THE PIPLELINE FIGURE -->

<!-- ![Primary multilayer perceptron architecture in the NeRF Uncertainty Ensemble]() -->

<!-- ![Secondary multilayer perceptron architecture in the NeRF Uncertainty Ensemble]() -->

## How to train NeRF Ensemble

### Run training!
<!-- TO BE WRITTEN -->

### Optional: Resume training from a checkpoint


Refer to [nerf-pytorch](https://github.com/krrish94/nerf-pytorch) for the detail of implementation and model training.

## Visualization uncertainty in NeRF Ensemble

<!-- TO BE WRITTEN -->

[NeRFDeltaView Ensemble](https://github.com/CTW121/NeRFDeltaView-Ensemble)
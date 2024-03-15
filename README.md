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

A neural radiance field (NeRF) is a simple fully connected network (weights are ~5MB) trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views.

Optimizing a NeRF takes between a few hours and a day or two (depending on resolution) and only requires a single GPU. Rendering an image from an optimized NeRF takes somewhere between less than a second and ~30 seconds, again depending on resolution.

## NeRF Ensemble

This NeRF Ensemble model is inspired by [Density-aware NeRF Ensemble](https://arxiv.org/abs/2209.08718) and [Deep Ensemble](https://arxiv.org/abs/1612.01474). This model estimates the NeRF model uncertainty based on estimated color $\boldsymbol{c}$ and volume density $\sigma$ from several multilayer perceptron (MLPs) with varying parameter initialization that captures epistemic uncertainty.

Following figure illustrates the Ensemble model pipeline.
![NeRF Uncertainty Ensemble pipeline](https://github.com/CTW121/NeRF-Uncertainty-Ensemble/blob/master/images/Ensemble_pipeline.png)

<!-- EXPLAIN THE PIPLELINE FIGURE -->

<!-- ![Primary multilayer perceptron architecture in the NeRF Uncertainty Ensemble]() -->

<!-- ![Secondary multilayer perceptron architecture in the NeRF Uncertainty Ensemble]() -->

After training the model, the uncertainty estimated by the model can be visualized using our uncertainty visualization tool, see section [Visualization uncertainty in NeRF Ensemble](##visualization-uncertainty-in-NeRF-Ensemble) for more information. 

## How to train NeRF Ensemble

### Run training!

First edit `config/chair.yml` to specify your own parameters.

The training script can be invoked by running
```bash
python train_nerf.py --config config/chair.yml
```

### Optional: Resume training from a checkpoint

Resume training from a previous checkpoint, run
```bash
python train_nerf.py --config config/chair.yml --load-checkpoint path/to/checkpoint.ckpt
```

Refer to [nerf-pytorch](https://github.com/krrish94/nerf-pytorch) for the detail of implementation and model training.

## Visualization uncertainty in NeRF Ensemble

The uncertainty estimated by the NeRF Ensemble model can be visualized using uncertainty visualization tool [NeRFDeltaView Ensemble](https://github.com/CTW121/NeRFDeltaView-Ensemble).

Uncertainty visualization provides users with an in-depth understanding of the data for analysis and to perform confident and informed decision-making. The main purpose of our tool is to highlight the significance of interactive visualization in enabling users to explore the estimated uncertainty in synthesized scenes, identify model limitations, and aid in understanding NeRF model uncertainty.
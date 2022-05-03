# PyTorchTraining.jl

This Julia package allows you to use and train PyTorch models with [FluxTraining.jl](https://github.com/FluxML/FluxTraining.jl). 

**This is a prototype. A more general approach to integrating PyTorch models in the FluxML ecosystem is being created at [PyCallChainRules.jl](https://github.com/rejuvyesh/PyCallChainRules.jl)**

## How do I use this?

You need to make 2 changes to your FluxTraining.jl setup:

1. add the `PyTorchBackend` callback
1. load a PyTorch `model` using PyCall.jl and pass it to `Learner`

```julia
using FluxTraining, PyCall, PyTorchTraining
model = PyCall.pyimport("torchvision").models.resnet18(pretrained=true)
learner = Learner(model, data, optim, lossfn, PyTorchBackend("cuda"))
```

See below for a full example of finetuning a pretrained vision model.

## Why should I use this?

This package could be useful for you if one or more of the following apply to you:

- you want to use pretrained PyTorch models
- you want to use research models published as PyTorch models
- you don't want to wait for your Julia model to compile ("Time To First Gradient"): no model compile times with this package
- you're fine with the standard ML use cases PyTorch covers
- you want to benefit from the hundreds of person-years PyTorch people have put into hyperoptimizing these standard use cases: lower memory usage and probably better performance since you can use larger batches
- you still want to use Julia for expensive preprocessing steps in your data pipeline

For an overview of the trade-offs in machine learning compilers that PyTorch and Flux.jl make, I suggest reading [Engineering Trade-Offs in Automatic Differentiation: from TensorFlow and PyTorch to Jax and Julia](https://www.stochasticlifestyle.com/engineering-trade-offs-in-automatic-differentiation-from-tensorflow-and-pytorch-to-jax-and-julia/) by Chris Rackauckas.


## Full example

This example gives the complete code for finetuning a pretrained image classifier from `torchvision` on the [Imagenette](https://github.com/fastai/imagenette) dataset. It uses [FastAI.jl](https://github.com/FluxML/FastAI.jl) for the data loading and preprocessing part.

```julia
using FastAI, FluxTraining, PyCall, PyTorchTraining
const torchvision = pyimport("torchvision")

function loadresnet(c::Int)
    # load pretrained resnet and replace last block with one outputting
    # `c` classes. Adapted from
    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    model = torchvision.models.resnet18(pretrained=true)
    model.fc = torch.nn.Linear(model.fc.in_features, c)
    return model
end


# Load dataset and create data loaders (FastAI.jl stuff)
data, blocks = loaddataset("imagenette2-320", (Image, Label))
method = ImageClassificationSingle(blocks, (224, 224))
dls = methoddataloaders(data, method, 64)
c = length(blocks[2].classes)

# Create a `Learner`
model = loadresnet(c)
learner = Learner(
    loadresnet(length(blocks[2].classes)),
    dls,
    ADAM(0.01),  # will be converted to `torch.optim.Adam`
    Flux.logitcrossentropy,
    PyTorchBackend()  # uses device "cuda" if available, else "cpu"
)

# Train it!

fit!(learner, 10)
```

## Caveats and nice-to-knows

This package...

- Uses [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) to use the Python `torch` library, but I haven't included a build step for installing it yet, so you'll have to set that up yourself for now.
- Tries to translate Flux optimisers and loss functions to their PyTorch equivalents, but I haven't implemented most of these translations. See `optimisers.jl` and `loss.jl` for how it's done. You can either pass in loss functions that work on tensors and PyTorch optimisers directly to `Learner` or implement the methods as in the source.
- Does not require using a custom training loop (everything is done via a callback) so it should work well for custom trainig loops like the one in [this VAE tutorial](https://fluxml.ai/FastAI.jl/dev/notebooks/vae.ipynb.html). Also see `callback.jl` if you want to see how it's done.
- automatically permutes dimensions of Julia arrays before converting them to PyTorch tensors since PyTorch uses different conventions for array dimension ordering. For example the batch dimension is last in Julia, but first in PyTorch and the image channel dimension comes after the spatial dimensions in Julia but before in PyTorch. The current default rules for how PyTorchTraining.jl decides how to permute are a bit ad-hoc but extensible. See `tensor.jl`.

# Features
 
Below a non-exhaustive list of features I have yet to add but will:

- hyperparameter scheduling on `PyTorchOptimiser`s
- compatibility with `Metric`s (without unnecessary copying)

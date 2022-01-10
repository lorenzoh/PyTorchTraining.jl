"""
Training loops on PyTorch models
"""


md"""

Is it possible to do everything via callback?

Only thing user would have to do is:
- pass in a `PyObject` model
- bass in a `PyTorchBackend(; device)` callback
- (optionally) pass in `PyObject` optimiser and loss functions (these can be
converted automatically)

The callback would do:

- `Zygote.gradient(::PyTorchParams)` should be overwritten to not calculate
    gradients, but just return the loss
- `BackwardBegin` should call `state.loss.backward()`
- `BackwardEnd` should set `state.loss = state.loss.item()` so that callbacks like
    `Metrics` will just see a number and it can be GC'd on Python side.
- `Flux.Optimise.update!(::PyTorchOptimiser, ::PyTorchParams, _)` should be overloaded
    to call `optim.step()` and `optim.zero_grad()`
"""


"""
    PyTorchBackend()

Callback that allows you to train PyTorch models with FluxTraining.jl.
To use, pass this callback to `Learner` and pass a `PyObject` PyTorch `model`.
"""
struct PyTorchBackend <: FluxTraining.Callback
    device::Any
end

PyTorchBackend() = PyTorchBackend(torch.cuda.is_available() ? "cuda" : "cpu")

import FluxTraining: init!, on, stateaccess, Read, Write, resolveconflict, Unresolvable

# During `init!`, the `Learner`'s `model`, `params`, `optimizer` and `lossfn`
# are, where possible, replaced with PyTorch versions.
function FluxTraining.init!(cb::PyTorchBackend, learner)
    learner.model = convertmodel(learner.model; device = cb.device)
    learner.params = Flux.params(learner.model)
    learner.optimizer = convertoptimiser(learner.optimizer, learner.params)
    learner.lossfn = convertlossfunction(learner.lossfn)
    return nothing
end

stateaccess(::PyTorchBackend) = (;
    model = Write(),
    params = Write(),
    optimizer = Write(),
    lossfn = Write(),
    step = Write(),
)

# `PyTorchBackend` should not be used with `ToGPU`/`ToDevice`:
resolveconflict(::PyTorchBackend, ::ToDevice) = Unresolvable()
resolveconflict(::ToDevice, ::PyTorchBackend) = Unresolvable()

function FluxTraining.on(::EpochBegin, ::AbstractTrainingPhase, cb::PyTorchBackend, learner)
    Flux.trainmode!(learner.model)
    learner.params = Flux.params(learner.model)
    learner.optimizer = convertoptimiser(learner.optimizer, learner.params)
end
function FluxTraining.on(::EpochBegin, ::AbstractValidationPhase, cb::PyTorchBackend, learner)
    Flux.testmode!(learner.model)
end

# At the beginning of every step, convert data to tensors on the correct
# device:
function FluxTraining.on(::StepBegin, ::Phase, cb::PyTorchBackend, learner)
    learner.step.xs = totensor(learner.step.xs, device=cb.device)
    learner.step.ys = totensor(learner.step.ys, device=cb.device)
end


# To make PyTorch training work with the default training loop, we'll
# implement a custom method for `Zygote.gradient` that dispatches on
# `PyTorchParams`.
Zygote.gradient(f, ::PyTorchParams) = f()


# Instead, when the `BackwardBegin` event is triggered, `loss.backward()`
# is called to do the backward pass. Importantly, this is done only during
# `AbstractTrainingPhase`s.
function FluxTraining.on(
    ::BackwardBegin,
    ::AbstractTrainingPhase,
    ::PyTorchBackend,
    learner,
)
    learner.step.loss.backward()
    learner.step.loss = learner.step.loss.item()
    # no gradient object to return
    return nothing
end


# On `StepEnd` during `AbstractValidationPhase`s, `learner.state.loss` is
# replaced by its scalar value so that callbacks like `Metrics` and can
# read it and things can be garbagecollected on the Python side. For
# `AbstractTrainingPhase`s this already happens during ´BackwardBegin`.

function FluxTraining.on(
    ::StepEnd,
    ::AbstractValidationPhase,
    ::PyTorchBackend,
    learner,
)
    learner.step.loss = learner.step.loss.item()
end


# Finally, `Flux.update!` is overloaded for `PyTorchOptimiser`

function Flux.Optimise.update!(o::PyTorchOptimiser, ::PyTorchParams, gs)
    o.optim.step()
    o.optim.zero_grad()
end

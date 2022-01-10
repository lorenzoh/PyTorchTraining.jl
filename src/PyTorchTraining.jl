module PyTorchTraining

using Flux
using FluxTraining, FluxTraining.Events, FluxTraining.Phases
using FluxTraining: runstep, runepoch
using InlineTest
using PyCall
using Markdown
import Zygote

include("tensor.jl")
include("model.jl")
include("optimisers.jl")
include("loss.jl")

include("callback.jl")

const torch = PyNULL()

function __init__()
    copy!(torch, pyimport("torch"))
end

export PyTorchBackend, totensor, fromtensor

end

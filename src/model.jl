#struct PyTorchModel end

mutable struct PyTorchModel
    model::PyObject
    grad::Bool
end
PyTorchModel(m) = PyTorchModel(m, true)
convertmodel(m::PyTorchModel; device=nothing) = m
convertmodel(p::PyObject; device=nothing) = PyTorchModel(p.to(device))

struct PyTorchParams
    params::PyObject
end

function (m::PyTorchModel)(args...; kwargs...)
    if m.grad
        return m.model(args..., kwargs...)
    else
        @pywith torch.no_grad() begin
            return m.model(args..., kwargs...)
        end
    end
end

function Flux.params(m::PyTorchModel)
    return PyTorchParams(pybuiltin("list")(m.model.parameters()))
end


function Flux.trainmode!(m::PyTorchModel, mode = true)
    m.grad = mode
    m.model = m.model.train(mode)
end
function Flux.testmode!(m::PyTorchModel, mode = true)
    Flux.trainmode!(m, !mode)
end

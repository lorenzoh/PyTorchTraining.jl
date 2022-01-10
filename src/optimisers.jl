"""
Functionality for PyTorch optimisers.
"""


struct PyTorchOptimiser
    optim
end

PyTorchOptimiser(opt::ADAM, ps::PyTorchParams) =
    PyTorchOptimiser(torch.optim.Adam(ps.params, lr=opt.eta, betas=opt.beta))

convertoptimiser(opt::Flux.Optimise.AbstractOptimiser, params) = PyTorchOptimiser(opt, params)
convertoptimiser(opt::PyTorchOptimiser, _) = opt

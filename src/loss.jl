function convertlossfunction(lossfn)
    error("""Cannot convert Julia loss function $lossfn to PyTorch-compatible
    loss function! Implement a method `convertlossfunction(::$(typeof(lossfn)))`.""")
end

convertlossfunction(lossfn::PyObject) = lossfn
convertlossfunction(::typeof(Flux.Losses.mse)) = torch.nn.functional.mse_loss
convertlossfunction(::typeof(Flux.Losses.logitcrossentropy)) = torch.nn.functional.mse_loss

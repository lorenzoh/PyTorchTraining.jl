function convertlossfunction(lossfn)
    error("""Cannot convert Julia loss function $lossfn to PyTorch-compatible
    loss function! Implement a method `convertlossfunction(::$(typeof(lossfn)))`.""")
end

convertlossfunction(lossfn::PyObject) = lossfn
convertlossfunction(::typeof(Flux.Losses.mse)) = torch.nn.functional.mse_loss


pylogitcrossentropy(ypreds, ys) =
    torch.nn.functional.cross_entropy(ypreds, ys.argmax(dim=1))

convertlossfunction(::typeof(Flux.logitcrossentropy)) = pylogitcrossentropy
convertlossfunction(l::typeof(pylogitcrossentropy)) = l

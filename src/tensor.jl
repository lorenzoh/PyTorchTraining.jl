"""
Convert Julia arrays to and from PyTorch `Tensor`s. Dimension ordering.

What do arrays/tensors look like?

- In Julia, the batch dimension is the LAST, in PyTorch is FIRST.
- In Julia, the channel dimensions comes AFTER the spatial dimensions,
    in PyTorch it comes BEFORE.

Example permutations:

- Batch of images: HWCB -> BCHW (4,3,1,2)
- Batch of onehot vectors: CB -> BC (2,1)

"""

abstract type ArrayFormat end


struct BatchFormat{F<:ArrayFormat} <: ArrayFormat
    format::F
end

BatchFormat() = BatchFormat(SameFormat())

struct SameFormat <: ArrayFormat end
struct ImageFormat <: ArrayFormat end

tensorpermutation(f::BatchFormat, n::Int) = (n, (tensorpermutation(f.format, n-1))...)
tensorpermutation(::SameFormat, n::Int) = ntuple(identity, n)
tensorpermutation(::ImageFormat, n::Int) = (n, ntuple(identity, n-1)...)

defaultformat(::AbstractArray) = BatchFormat()
defaultformat(::AbstractArray{T, 4}) where T = BatchFormat(ImageFormat())


function totensor(a::AbstractArray, format = defaultformat(a); kwargs...)
    return torch.tensor(permutedims(a, tensorpermutation(format, ndims(a))); kwargs...)
end

totensor(t::PyObject, args...; kwargs...) = t


function fromtensor(a::PyObject, format = defaultformat(a))
    return pycall(a.cpu().numpy, PyArray)
end

@testset "tensorpermutation" begin
    @test tensorpermutation(BatchFormat(), 4) == (4, 1, 2, 3)
    @test tensorpermutation(SameFormat(), 4) == (1, 2, 3, 4)
    @test tensorpermutation(ImageFormat(), 3) == (3, 1, 2)
end

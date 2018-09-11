module GPUArrays

export GPUArray, gpu_call, thread_blocks_heuristic, global_size, synchronize_threads
export linear_index, @linearidx, @cartesianidx, convolution!, device, synchronize
export JLArray

using Serialization
using Random
using LinearAlgebra
using Printf

using LinearAlgebra.BLAS
using Base.Cartesian

using FFTW

using SimpleTraits
using MacroTools
using InteractiveUtils

include("abstractarray.jl")
include("abstract_gpu_interface.jl")
include("ondevice.jl")
include("base.jl")
include("construction.jl")
include("blas.jl")

include("devices.jl")
include("heuristics.jl")
include("indexing.jl")
include("linalg.jl")
include("mapreduce.jl")
include("vectors.jl")
include("convolution.jl")
include("random.jl")
include("wrapper.jl")
include("broadcast.jl")

include("array.jl")

include("testsuite.jl")

end # module

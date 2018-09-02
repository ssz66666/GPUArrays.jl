# Very simple Julia back-end which is just for testing the implementation and can be used as
# a reference implementation


## construction

struct JLArray{T, N} <: GPUArray{T, N}
    data::Array{T, N}
    size::Dims{N}

    function JLArray{T,N}(data::Array{T, N}, size::NTuple{N, Int}) where {T,N}
        new(data, size)
    end
end

JLArray(data::AbstractArray{T, N}, size::Dims{N}) where {T,N} = JLArray{T,N}(data, size)

(::Type{<: JLArray{T}})(x::AbstractArray) where T = JLArray(convert(Array{T}, x), size(x))

function JLArray{T, N}(size::NTuple{N, Integer}) where {T, N}
    JLArray{T, N}(Array{T, N}(undef, size), size)
end

struct JLBackend <: GPUBackend end
_backend(::Type{<:JLArray}) = JLBackend()

## getters

Base.size(x::JLArray) = x.size

Base.pointer(x::JLArray) = pointer(x.data)


## other

"""
Thread group local memory
"""
struct LocalMem{N, T}
    x::NTuple{N, Vector{T}}
end

to_device(state, x::JLArray) = x.data
to_device(state, x::Tuple) = to_device.(Ref(state), x)
to_device(state, x::Base.RefValue{<: JLArray}) = Base.RefValue(to_device(state, x[]))
to_device(state, x) = x
# creates a `local` vector for each thread group
to_device(state, x::LocalMemory{T}) where T = LocalMem(ntuple(i-> Vector{T}(x.size), blockdim_x(state)))

to_blocks(state, x) = x
# unpacks local memory for each block
to_blocks(state, x::LocalMem) = x.x[blockidx_x(state)]

Base.similar(::Type{<: JLArray}, ::Type{T}, size::Base.Dims{N}) where {T, N} = JLArray{T, N}(size)

unsafe_reinterpret(::Type{T}, A::JLArray, size::Tuple) where T =
    reshape(reinterpret(T, A.data), size)

function Base.unsafe_copyto!(dest::Array{T}, d_offset::Integer,
                             source::JLArray{T}, s_offset::Integer,
                             amount::Integer) where T
    copyto!(dest, d_offset, source.data, s_offset, amount)
end
function Base.unsafe_copyto!(dest::JLArray{T}, d_offset::Integer,
                             source::Array{T}, s_offset::Integer,
                             amount::Integer) where T
    copyto!(dest.data, d_offset, source, s_offset, amount)
    dest
end
function Base.unsafe_copyto!(dest::JLArray{T}, d_offset::Integer,
                             source::JLArray{T}, s_offset::Integer,
                             amount::Integer) where T
    copyto!(dest.data, d_offset, source.data, s_offset, amount)
    dest
end

mutable struct JLState{N}
    blockdim::NTuple{N, Int}
    griddim::NTuple{N, Int}

    blockidx::NTuple{N, Int}
    threadidx::NTuple{N, Int}
    localmem_counter::Int
    localmems::Vector{Vector{Array}}
end

function JLState(threads::NTuple{N}, blockdim::NTuple{N}) where N
    idx = ntuple(i-> 1, Val(N))
    blockcount = prod(blockdim)
    lmems = [Vector{Array}() for i in 1:blockcount]
    JLState{N}(threads, blockdim, idx, idx, 0, lmems)
end

function JLState(state::JLState{N}, threadidx::NTuple{N}) where N
    JLState{N}(
        state.blockdim,
        state.griddim,
        state.blockidx,
        threadidx,
        0,
        state.localmems
    )
end

function LocalMemory(state::JLState, ::Type{T}, ::Val{N}, ::Val{C}) where {T, N, C}
    state.localmem_counter += 1
    lmems = state.localmems[blockidx_x(state)]
    # first invokation in block
    if length(lmems) < state.localmem_counter
        lmem = fill(zero(T), N)
        push!(lmems, lmem)
        return lmem
    else
        return lmems[state.localmem_counter]
    end
end

function AbstractDeviceArray(ptr::Array, shape::NTuple{N, Integer}) where N
    reshape(ptr, shape)
end
function AbstractDeviceArray(ptr::Array, shape::Vararg{Integer, N}) where N
    reshape(ptr, shape)
end

function _gpu_call(::JLBackend, f, A, args::Tuple, blocks_threads::Tuple{T, T}) where T <: NTuple{N, Integer} where N
    blocks, threads = blocks_threads
    idx = ntuple(i-> 1, length(blocks))
    blockdim = blocks
    state = JLState(threads, blockdim)
    device_args = to_device.(Ref(state), args)
    tasks = Array{Task}(undef, threads...)
    for blockidx in CartesianIndices(blockdim)
        state.blockidx = blockidx.I
        block_args = to_blocks.(Ref(state), device_args)
        for threadidx in CartesianIndices(threads)
            thread_state = JLState(state, threadidx.I)
            tasks[threadidx] = @async @allowscalar f(thread_state, block_args...)
            # TODO: @async obfuscates the trace to any exception which happens during f
        end
        for t in tasks
            fetch(t)
        end
    end
    return
end

# "intrinsics"
struct JLDevice end
device(x::JLArray) = JLDevice()
threads(dev::JLDevice) = 256
blocks(dev::JLDevice) = (256, 256, 256)

@inline function synchronize_threads(::JLState)
    #=
    All threads are getting started asynchronously,so a yield will
    yield to the next execution of the same function, which should call yield
    at the exact same point in the program, leading to a chain of yields  effectively syncing
    the tasks (threads).
    =#
    yield()
    return
end

for (i, sym) in enumerate((:x, :y, :z))
    for f in (:blockidx, :blockdim, :threadidx, :griddim)
        fname = Symbol(string(f, '_', sym))
        @eval $fname(state::JLState) = Int(state.$f[$i])
    end
end

blas_module(::JLArray) = LinearAlgebra.BLAS
blasbuffer(A::JLArray) = A.data

# defining our own plan type is the easiest way to pass around the plans in FFTW interface
# without ambiguities

struct FFTPlan{T}
    p::T
end

FFTW.plan_fft(A::JLArray; kw_args...) = FFTPlan(plan_fft(A.data; kw_args...))
FFTW.plan_fft!(A::JLArray; kw_args...) = FFTPlan(plan_fft!(A.data; kw_args...))
FFTW.plan_bfft!(A::JLArray; kw_args...) = FFTPlan(plan_bfft!(A.data; kw_args...))
FFTW.plan_bfft(A::JLArray; kw_args...) = FFTPlan(plan_bfft(A.data; kw_args...))
FFTW.plan_ifft!(A::JLArray; kw_args...) = FFTPlan(plan_ifft!(A.data; kw_args...))
FFTW.plan_ifft(A::JLArray; kw_args...) = FFTPlan(plan_ifft(A.data; kw_args...))

function Base.:(*)(plan::FFTPlan, A::JLArray)
    x = plan.p * A.data
    JLArray(x)
end

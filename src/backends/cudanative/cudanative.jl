module CUBackend

using ..GPUArrays, CUDAnative, StaticArrays

import CUDAdrv, CUDArt #, CUFFT

import GPUArrays: buffer, create_buffer, acc_mapreduce, is_cudanative
import GPUArrays: Context, GPUArray, context, linear_index, gpu_call, free_global_memory
import GPUArrays: blas_module, blasbuffer, is_blas_supported, hasblas, init
import GPUArrays: default_buffer_type, broadcast_index, is_fft_supported, unsafe_reinterpret
import GPUArrays: is_gpu, name, threads, blocks, global_memory, local_memory, new_context
using GPUArrays: device_summary

using CUDAdrv: CuDefaultStream

immutable GraphicsResource{T}
    glbuffer::T
    resource::Ref{CUDArt.rt.cudaGraphicsResource_t}
    ismapped::Bool
end

immutable CUContext <: Context
    ctx::CUDAdrv.CuContext
    device::CUDAdrv.CuDevice
end
is_cudanative(ctx::CUContext) = true
function Base.show(io::IO, ctx::CUContext)
    println(io, "CUDAnative context with:")
    device_summary(io, ctx.device)
end


devices() = CUDAdrv.devices()
is_gpu(dev::CUDAdrv.CuDevice) = true
name(dev::CUDAdrv.CuDevice) = string("CU ", CUDAdrv.name(dev))
threads(dev::CUDAdrv.CuDevice) = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)

function blocks(dev::CUDAdrv.CuDevice)
    (
        CUDAdrv.attribute(dev, CUDAdrv.MAX_BLOCK_DIM_X),
        CUDAdrv.attribute(dev, CUDAdrv.MAX_BLOCK_DIM_Y),
        CUDAdrv.attribute(dev, CUDAdrv.MAX_BLOCK_DIM_Z),
    )
end

free_global_memory(dev::CUDAdrv.CuDevice) = CUDAdrv.Mem.info()[1]
global_memory(dev::CUDAdrv.CuDevice) = CUDAdrv.totalmem(dev)
local_memory(dev::CUDAdrv.CuDevice) = CUDAdrv.attribute(dev, CUDAdrv.TOTAL_CONSTANT_MEMORY)


#const GLArrayImg{T, N} = GPUArray{T, N, gl.Texture{T, N}, GLContext}
const CUArray{T, N, B} = GPUArray{T, N, B, CUContext} #, GLArrayImg{T, N}}
const CUArrayBuff{T, N} = CUArray{T, N, CUDAdrv.CuArray{T, N}}


global all_contexts, current_context, current_device

let contexts = Dict{CUDAdrv.CuDevice, CUContext}(), active_device = CUDAdrv.CuDevice[]

    all_contexts() = values(contexts)
    function current_device()
        if isempty(active_device)
            push!(active_device, CUDAnative.default_device[])
        end
        active_device[]
    end
    function current_context()
        dev = current_device()
        get!(contexts, dev) do
            new_context(dev)
        end
    end

    function GPUArrays.init(dev::CUDAdrv.CuDevice)
        GPUArrays.setbackend!(CUBackend)
        if isempty(active_device)
            push!(active_device, dev)
        else
            active_device[] = dev
        end
        ctx = get!(()-> new_context(dev), contexts, dev)
        CUDAdrv.activate(ctx.ctx)
        ctx
    end

    function GPUArrays.destroy!(context::CUContext)
        # don't destroy primary device context
        dev = context.device
        if haskey(contexts, dev) && contexts[dev] == context
            error("Trying to destroy primary device context which is prohibited. Please use reset!(context)")
        end
        CUDAdrv.destroy!(context.ctx)
        return
    end
end

function reset!(context::CUContext)
    dev = context.device
    CUDAdrv.destroy!(context.ctx)
    context.ctx = CUDAdrv.CuContext(dev)
    return
end

function new_context(dev::CUDAdrv.CuDevice)
    cuctx = CUDAdrv.CuContext(dev)
    ctx = CUContext(cuctx, dev)
    CUDAdrv.activate(cuctx)
    return ctx
end


# synchronize
function GPUArrays.synchronize{T, N}(x::CUArray{T, N})
    CUDAdrv.synchronize(context(x).ctx) # TODO figure out the diverse ways of synchronization
end

function GPUArrays.free{T, N}(x::CUArray{T, N})
    GPUArrays.synchronize(x)
    Base.finalize(buffer(x))
    nothing
end


default_buffer_type{T, N}(::Type, ::Type{Tuple{T, N}}, ::CUContext) = CUDAdrv.CuArray{T, N}

function (AT::Type{CUArray{T, N, Buffer}}){T, N, Buffer <: CUDAdrv.CuArray}(
        size::NTuple{N, Int};
        context = current_context(),
        kw_args...
    )
    # cuda doesn't allow a size of 0, but since the length of the underlying buffer
    # doesn't matter, with can just initilize it to 0
    buff = prod(size) == 0 ? CUDAdrv.CuArray{T}((1,)) : CUDAdrv.CuArray{T}(size)
    AT(buff, size, context)
end

function unsafe_reinterpret(::Type{T}, A::CUArray{ET}, dims::NTuple{N, Integer}) where {T, ET, N}
    buff = buffer(A)
    newbuff = CUDAdrv.CuArray{T, N}(dims, convert(CUDAdrv.OwnedPtr{T}, pointer(buff)))
    ctx = context(A)
    GPUArray{T, length(dims), typeof(newbuff), typeof(ctx)}(newbuff, dims, ctx)
end


function Base.copy!{T}(
        dest::Array{T}, d_offset::Integer,
        source::CUDAdrv.CuArray{T}, s_offset::Integer, amount::Integer
    )
    amount == 0 && return dest
    d_offset = d_offset
    s_offset = s_offset - 1
    device_ptr = pointer(source)
    sptr = device_ptr + (sizeof(T) * s_offset)
    CUDAdrv.Mem.download(Ref(dest, d_offset), sptr, sizeof(T) * (amount))
    dest
end
function Base.copy!{T}(
        dest::CUDAdrv.CuArray{T}, d_offset::Integer,
        source::Array{T}, s_offset::Integer, amount::Integer
    )
    amount == 0 && return dest
    d_offset = d_offset - 1
    s_offset = s_offset
    device_ptr = pointer(dest)
    sptr = device_ptr.ptr + (sizeof(T) * d_offset)
    CUDAdrv.Mem.upload(sptr, Ref(source, s_offset), sizeof(T) * (amount))
    dest
end


function Base.copy!{T}(
        dest::CUDAdrv.CuArray{T}, d_offset::Integer,
        source::CUDAdrv.CuArray{T}, s_offset::Integer, amount::Integer
    )
    d_offset = d_offset - 1
    s_offset = s_offset - 1
    d_ptr = pointer(dest)
    s_ptr = pointer(source)
    dptr = CUDAdrv.OwnedPtr{T}(d_ptr.ptr + (sizeof(T) * d_offset), d_ptr.ctx)
    sptr = CUDAdrv.OwnedPtr{T}(s_ptr.ptr + (sizeof(T) * s_offset), s_ptr.ctx)
    CUDAdrv.Mem.transfer(sptr, dptr, sizeof(T) * (amount))
    dest
end

function thread_blocks_heuristic(A::AbstractArray)
    thread_blocks_heuristic(length(A))
end

thread_blocks_heuristic{N}(s::NTuple{N, Integer}) = thread_blocks_heuristic(prod(s))
function thread_blocks_heuristic(len::Integer)
    threads = min(len, 256)
    blocks = ceil(Int, len/threads)
    blocks, threads
end

@inline function linear_index(::CUDAnative.CuDeviceArray, state)
    Cuint((blockIdx().x - Cuint(1)) * blockDim().x + threadIdx().x)
end


unpack_cu_array(x) = x
unpack_cu_array(x::Scalar) = unpack_cu_array(getfield(x, 1))
unpack_cu_array{T,N}(x::CUArray{T,N}) = buffer(x)
unpack_cu_array(x::Ref{<:GPUArrays.AbstractAccArray}) = unpack_cu_array(x[])

# TODO hook up propperly with CUDAdrv... This is a dirty adhoc solution
# to be consistent with the OpenCL backend
immutable CUFunction{T}
    kernel::T
end

hasnvcc() = true


function CUFunction{T, N}(A::CUArray{T, N}, f::Function, args...)
    CUFunction(f) # this is mainly for consistency with OpenCL
end
function CUFunction{T, N}(A::CUArray{T, N}, f::Tuple{String, Symbol}, args...)
    source, name = f
    kernel_name = string(name)
    ctx = context(A)
    kernel = CUDArt._compile(ctx.device, kernel_name, source, "from string")
    CUFunction(kernel) # this is mainly for consistency with OpenCL
end
function (f::CUFunction{F}){F <: Function, T, N}(A::CUArray{T, N}, args...)
    dims = thread_blocks_heuristic(A)
    return CUDAnative.generated_cuda(
        dims, 0, CuDefaultStream(),
        f.kernel, map(unpack_cu_array, args)...
    )
end

cudacall_types(x::CUArray{T, N}) where {T, N} = Ptr{T}
cudacall_types(x::T) where T = T

cudacall_convert(x) = x
cudacall_convert(x::CUArray{T, N}) where {T, N} = pointer(buffer(x))

function (f::CUFunction{F}){F <: CUDAdrv.CuFunction, T, N}(A::CUArray{T, N}, args)
    griddim, blockdim = thread_blocks_heuristic(A)
    typs = Tuple{cudacall_types.(args)...}
    cuargs = cudacall_convert.(args)
    CUDAdrv.cudacall(
        f.kernel, CUDAdrv.CuDim3(griddim...), CUDAdrv.CuDim3(blockdim...),
        typs, cuargs...
    )
end

function gpu_call{T, N}(f::Function, A::CUArray{T, N}, args::Tuple, globalsize = length(A), localsize = nothing)
    blocks, thread = thread_blocks_heuristic(globalsize)
    args = map(unpack_cu_array, args)
    #cu_kernel, rewritten = CUDAnative.rewrite_for_cudanative(kernel, map(typeof, args))
    #println(CUDAnative.@code_typed kernel(args...))
    @cuda (blocks, thread) f(0f0, args...)
end
function gpu_call{T, N}(f::Tuple{String, Symbol}, A::CUArray{T, N}, args::Tuple, globalsize = size(A), localsize = nothing)
    func = CUFunction(A, f, args...)
    # TODO cache
    func(A, args) # TODO pass through local/global size
end

#####################################
# The problem is, that I can't pass Tuple{CuArray} as a type, so I can't
# write a @generated function to unrole the arguments.
# And without unroling of the arguments, GPU codegen will cry!

for i = 0:10
    args = ntuple(x-> Symbol("arg_", x), i)
    fargs2 = ntuple(x-> :(broadcast_index($(args[x]), sz, i)), i)
    @eval begin

        function reduce_kernel{F <: Function, OP <: Function,T1, T2, N}(
                out::AbstractArray{T2,1}, f::F, op::OP, v0::T2,
                A::AbstractArray{T1, N}, $(args...)
            )
            #reduce multiple elements per thread

            i = (CUDAnative.blockIdx().x - UInt32(1)) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
            step = CUDAnative.blockDim().x * CUDAnative.gridDim().x
            sz = Cuint.(size(A))
            result = v0
            while i <= length(A)
                @inbounds result = op(result, f(A[i], $(fargs2...)))
                i += step
            end
            result = reduce_block(result, op, v0)
            if CUDAnative.threadIdx().x == UInt32(1)
                @inbounds out[CUDAnative.blockIdx().x] = result
            end
            return
        end
    end
end


#################################
# Reduction

# TODO do this correctly in CUDAnative/Base
using ColorTypes

@generated function CUDAnative.shfl_down(
        val::T, srclane::Integer, width::Integer = Int32(32)
    ) where T
    constr = Expr(:new, T)
    for fname in fieldnames(T)
        push!(constr.args, :(CUDAnative.shfl_down(getfield(val, $(QuoteNode(fname))), srclane, width)))
    end
    constr
end

function reduce_warp{T, F<:Function}(val::T, op::F)
    offset = CUDAnative.warpsize() ÷ UInt32(2)
    while offset > UInt32(0)
        val = op(val, CUDAnative.shfl_down(val, offset))
        offset ÷= UInt32(2)
    end
    return val
end

@inline function reduce_block{T, F <: Function}(val::T, op::F, v0::T)::T
    shared = CUDAnative.@cuStaticSharedMem(T, 32)
    wid  = div(CUDAnative.threadIdx().x - UInt32(1), CUDAnative.warpsize()) + UInt32(1)
    lane = rem(CUDAnative.threadIdx().x - UInt32(1), CUDAnative.warpsize()) + UInt32(1)

     # each warp performs partial reduction
    val = reduce_warp(val, op)

    # write reduced value to shared memory
    if lane == 1
        @inbounds shared[wid] = val
    end
    # wait for all partial reductions
    CUDAnative.sync_threads()
    # read from shared memory only if that warp existed
    @inbounds begin
        val = (threadIdx().x <= fld(CUDAnative.blockDim().x, CUDAnative.warpsize())) ? shared[lane] : v0
    end
    if wid == 1
        # final reduce within first warp
        val = reduce_warp(val, op)
    end
    return val
end



function acc_mapreduce{T, OT, N}(
        f, op, v0::OT, A::CUArray{T, N}, rest::Tuple
    )
    dev = context(A).device
    @assert(CUDAdrv.capability(dev) >= v"3.0", "Current CUDA reduce implementation requires a newer GPU")
    threads = 256
    blocks = min((length(A) + threads - 1) ÷ threads, 1024)
    out = similar(buffer(A), OT, (blocks,))
    args = map(unpack_cu_array, rest)
    # TODO MAKE THIS WORK FOR ALL FUNCTIONS .... v0 is really unfit for parallel reduction
    # since every thread basically needs its own v0
    @cuda (blocks, threads) reduce_kernel(out, f, op, v0, buffer(A), args...)
    # for this size it doesn't seem beneficial to run on gpu?!
    # TODO actually benchmark this theory
    reduce(op, Array(out))
end


########################################
# CUBLAS

function to_cudart(A::CUArray)
    ctx = context(A)
    buff = buffer(A)
    devptr = pointer(buff)
    device = CUDAdrv.device(devptr.ctx).handle
    CUDArt.CudaArray(CUDArt.CudaPtr(devptr.ptr, ctx.ctx), size(A), Int(device))
end

if is_blas_supported(:CUBLAS)
    using CUBLAS
    import CUDArt
    #
    # # implement blas interface
    hasblas(::CUContext) = true
    blas_module(::CUContext) = CUBLAS
    blasbuffer(ctx::CUContext, A) = buffer(A)
end

if is_fft_supported(:CUFFT)
    include("fft.jl")
end

#
# function convert{T <: CUArray}(t::T, A::CUDArt.CudaArray)
#     ctx = context(t)
#     ptr = DevicePtr(context(t))
#     device = CUDAdrv.device(devptr.ctx).handle
#     CUDArt.CudaArray(CUDArt.CudaPtr(devptr.ptr), size(A), Int(device))
#     CuArray(size(A))
# end
#
#

export CUFunction

end

using .CUBackend
export CUBackend

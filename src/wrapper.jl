# TODO
# 1. using @gpu_type annotation, remember locations
# 2. code generation for fall backs (require invoke)
#    need to use macro to produce a generated function,
#    which will have access to the type informations
#    it would be great if we can figure this out
#    before compilation though

# trait defining GPU optimisable (wrapped) array types

@traitdef IsGPUOptimizable{A}

# marker macro used by gpu_override_method, must directly decorate a type
# that is a proper supertype of a GPU-compatible array type.
macro gpu_type(ex)
    ex
end

# generate trait dispatch functions for types marked with @gpu_type
# AbstractArray (without parameter) is mapped to GPUDestArray,
# arbitrary wrapper type T is mapped to T{<:GPUDestArray} (marking the parent type
# as GPUDestArray)
macro gpu_override_method(ex)
    
end

#const supported_wrappers = Set{Type}([
#    LinearAlgebra.Transpose,
#    LinearAlgebra.Adjoint
#])

# utility functions to manipulate types

_collect_union(t::Type,set) = push!(set,t)
_collect_union(t::Union,set) = _collect_union(t.b, push!(set,t.a))
collect_union(t) = _collect_union(t,Type[])

_unpack_abstract(x::Type) = isabstracttype(x) ? Union{map(_unpack_abstract,InteractiveUtils.subtypes(x))...} : x

# recursively update a type variable without modifying the input type
update_type_var(t,_) = t
update_type_var(t::TypeVar,tvtable) = begin
    if haskey(tvtable,t)
        tvtable[t]
    else
        t
    end
end
update_type_var(t::DataType,tvtable) = begin
    old_params = collect(t.parameters)
    new_params = map(p->update_type_var(p,tvtable),old_params)
    if old_params == new_params
        t
    else
        Core.apply_type(t.name.wrapper,new_params...)
    end
end
update_type_var(t::Core.TypeofBottom,_) = t
update_type_var(t::Union,tvtable) = begin
    old_union = collect_union(t)
    new_union = map(_t->update_type_var(_t,tvtable),old_union)
    if old_union == new_union
        t
    else
        Union{new_union...}
    end
end
update_type_var(t::UnionAll,tvtable) = begin
    _t = t
    tvs = TypeVar[]
    while _t isa UnionAll
        push!(tvs,_t.var)
        _t = _t.body
    end
    new_t = update_type_var(_t,tvtable)
    if _t == new_t
        t
    else
        while !isempty(tvs)
            _tv = pop!(tvs)
            new_t = UnionAll(update_type_var(_tv,tvtable),new_t)
        end
        new_t
    end
end

# update type variable, given information from tvtable
# we use this function to propagate changes to all type variables
# affected
update_type_var_ref(t::TypeVar,tvtable) = begin
    new_lb = update_type_var(t.lb,tvtable)
    new_ub = update_type_var(t.ub,tvtable)
    if (new_ub == t.ub) && (new_lb == t.lb)
        t
    else
        TypeVar(t.name,new_lb,new_ub)
    end
end

# turn AST into type/type variable declarations.
# resulting types can contain declared free variables
# these "pseudo" types are used to generate appropriate
# subtype for the annotated dispatch target(s)

# Macro are considered to be annotations for type/type vars, and stripped.
# Annotated type/type vars are collected in the returned `macros` dictionary
# e.g. all types/type vars annotated with `@gpu_type` will be in `macros[Symbol("@gpu_type")]`
# this certainly breaks some macros e.g. `@nospecialize` if not recovered properly

expr_to_pseudo_typevars(ts) = begin
    tvtable = [Dict()]
    macros = Dict()
    (map(t->expr_to_pseudo_typevar(t,tvtable,macros),ts),tvtable,macros)
end

expr_to_pseudo_typevar(t,tvtable,macros) = begin
    if t isa Expr && t.head == :macrocall
        ret = expr_to_pseudo_typevar(t.args[3],tvtable,macros)
        if !haskey(macros,t.args[1])
            push!(macros,t.args[1] => [])
        end
        push!(macros[t.args[1]],ret)
        ret
    else
        local v, tv
        @match t begin
            LB_ <: TV_ <: UB_ => begin
                v = TypeVar(TV,expr_to_pseudo_type(LB,tvtable,macros),expr_to_pseudo_type(UB,tvtable,macros))
                tv = TV
            end
            TV_ <: UB_ => begin 
                v = TypeVar(TV,expr_to_pseudo_type(UB,tvtable,macros))
                tv = TV
            end
            TV_ >: LB_ => begin 
                v = TypeVar(TV,expr_to_pseudo_type(LB,tvtable,macros),Any)
                tv = TV
            end
            TV_ => begin
                v = TypeVar(TV)
                tv = TV
            end
            _ => error("failed to parse type variable $t")
        end
        push!(tvtable[end],tv => v)
        v
    end
end
expr_to_pseudo_type_param(t,tvtable,macros) = begin
    if t isa Expr && t.head == :macrocall
        ret = expr_to_pseudo_type_param(t.args[3],tvtable,macros)
        if !haskey(macros,t.args[1])
            push!(macros,t.args[1] => [])
        end
        push!(macros[t.args[1]],ret)
        ret
    else
        @match t begin
            <: T_ => begin
                ub = expr_to_pseudo_type(T,tvtable,macros)
                tv = gensym()
                v = TypeVar(tv,ub)
                push!(tvtable[end],tv => v)
                v
            end
            >: T_ => begin
                lb = expr_to_pseudo_type(T,tvtable,macros)
                tv = gensym()
                v = TypeVar(tv,lb,Any)
                push!(tvtable[end],tv => v)
                v
            end
            _ => expr_to_pseudo_type(t,tvtable,macros)
        end
    end
end
expr_to_pseudo_type(t,tvtable,macros) = begin
    if t isa Expr && t.head == :macrocall
        ret = expr_to_pseudo_type(t.args[3],tvtable,macros)
        if !haskey(macros,t.args[1])
            push!(macros,t.args[1] => [])
        end
        push!(macros[t.args[1]],ret)
        ret
    else
        local _ret
        push!(tvtable,Dict())
        @match t begin
            T_ where TV__ => begin
                foreach(e->expr_to_pseudo_typevar(e,tvtable,macros),TV)
                _ret = expr_to_pseudo_type(T,tvtable,macros)
            end
            T_{PARAM__} => begin
                # T must be a type, not typevar
                typ = eval(T)
                params = map(p->expr_to_pseudo_type_param(p,tvtable,macros),PARAM)
                _ret = Core.apply_type(typ,params...)
            end
            T_ => begin
                isbound = false
                local ret
                for _i = length(tvtable):-1:1
                    if haskey(tvtable[_i],T)
                        isbound = true
                        ret = tvtable[_i][T]
                        break
                    end
                end
                if isbound
                    _ret = ret
                else
                    _ret = eval(T)
                end
            end
        end
        for (_, v) in tvtable[end]
            _ret = UnionAll(v,_ret)
        end
        pop!(tvtable)
        _ret
    end
end

# atoms e.g. numbers, symbol literals
expr_from_type_or_typevar(t) = expr_from_type(t)
expr_from_type_or_typevar(t::TypeVar) = :($(expr_from_type(t.lb)) <: $(t.name) <: $(expr_from_type(t.ub)) )
expr_from_type(t) = t
expr_from_type(t::TypeVar) = t.name
# free variables are assumed bound outside
expr_from_type(t::DataType) = begin
    if length(t.parameters) == 0
        Symbol(t.name)
    else
        Expr(
            :curly,
            Symbol(t.name),
            map(expr_from_type,t.parameters)...
        )
    end
end
expr_from_type(t::Core.TypeofBottom) = :(Union{})
expr_from_type(t::Union) = Expr(:curly,:Union,map(collect_union(t))...)
expr_from_type(t::UnionAll) = Expr(:where, expr_from_type(t.body), expr_from_type_or_typevar(t.var))

# push UnionAlls to the bottom to make finding type params of Union type easier
_flatten_union(t::UnionAll,tvs) = _flatten_union(t.body,push!(tvs,t.var))
_flatten_union(t::Union,tvs) = Union{map(x->_flatten_union(x,tvs),collect_union(t))...}
_flatten_union(t,tvs) = begin
    for i in length(tvs):-1:1
        t = UnionAll(tvs[i],t)
    end
    t
end
flatten_union(t) = _flatten_union(t,[])

const supported_wrappers = collect_union(
    Union{map(flatten_union,
    (map(_unpack_abstract,
    filter(t->t!=AbstractArray,
    map(m->m.sig.parameters[2],
    methods(Base.parent))))))...}
)

# dummy type variable type to help guess the type parameter representing
# the parent array type
struct _TV{M} end

_unwind_unionall(t) = Base.unwrap_unionall(t)
_unwind_unionall(t::Type{T}, n) where {T} = (t, n)
_unwind_unionall(t::UnionAll, n) = _unwind_unionall(t.body, n+1)

_rewind_unionall!(t, tvs) = begin
    while !(isempty(tvs))
        t = UnionAll(pop!(tvs), t)
    end
    t
end

for wp in supported_wrappers

    # get number of parameters of UnionAll type
    local nparam = last(_unwind_unionall(wp,0))
    # populate the UnionAll by applying type vars with upper bound  _TV{i}
    # e.g. SubArray{T1,T2,T3,T4,T5} where {T1<:_TV{1},T2<:_TV{2},T3<:_TV{3},T4<:_TV{4},T5<:_TV{5}}
    local typevs = [TypeVar(Symbol("T$i"),Union{},_TV{i}) for i = 1:nparam]
    local populated = Core.apply_type(wp,typevs...)
    # rewrap the populated type back to a UnionAll
    local rewrapped = _rewind_unionall!(populated, typevs)
    local m = first(methods(Base.parent,[wp]))

    # figure out the relationship between the parent type and the type parameters
    # here we do a type inference on Base.parent to guess which type parameter of the wrapper
    # represents the parent array type. 
    local pt = Core.Compiler.typeinf_type(m,Tuple{rewrapped},Core.svec(),Core.Compiler.Params(UInt(m.min_world)))
    if pt <: _TV
        # success!
        local n = first(pt.parameters)
        @eval parent_type_param_pos(::Type{<:$wp}) = $n
    
        # used by other macro calls to generate appropriate signatures
        # to allow overriding functions directly dispatching on wrapper types
        
        # generate functions of the form:
        #   parent_type(::Type{<:wp{[(n - 1) * <:Any] , T}}) = T
 
        @eval parent_type(::Type{<:$(Expr(
            :curly,
            :($wp),
            fill(:(<:Any),n-1)...,
            :PT
        ))}) where PT = PT

        
        @eval _trecurse(::Type{<:$(Expr(
            :curly,
            :($wp),
            fill(:(<:Any),n-1)...,
            :PT
        ))}) where PT = _trecurse(PT)
        

    else
        # we can't handle it automatically, abort
        pop!(supported_wrappers,wp)
    end
end

# produce a hacky Union that is large enough to cover all wrapper types we support
# this has to be smaller than AbstractArray
# we will use trait function to identify and redirect incorrectly overridden functions
# to the fall back
const GPUDestArray = Union{GPUArray,supported_wrappers...}

# we replace annotated types with a specialised version

# resolve conflict assignments to the same type variable

function _combine_tv_replacement(dict1,dict2)
    if isempty(dict1)
        dict2
    else
        for (k, v) in dict2
            if haskey(dict1,k)
                dict1[k] = typeintersect(dict1[k],v)
            else
                dict1[k] = v
            end
        end
        dict1
    end
end

# collect type variable dependency information
function type_var_dependency(tvs)
    nodes = Set{TypeVar}()
    fedges = Dict{TypeVar,Set{TypeVar}}() # A => [B] A affects B
    bedges = Dict{TypeVar,Set{TypeVar}}() # B => [A] B depends on A
    foreach(_tv->_type_var_dependency(_tv,nodes,fedges,bedges),tvs)
    nodes, fedges, bedges
end

function _type_var_dependency(tv::TypeVar,nodes,forward_edges,backward_edges)
    if !(tv in nodes)
        push!(nodes, tv)
        _lb = _type_var_dependency(tv.lb,nodes,edges)
        _ub = _type_var_dependency(tv.ub,nodes,edges)
        _tvs = union(_lb,_ub)
        push!(backward_edges,tv => Set{TypeVar}())
        for v in _tvs
            haskey(forward_edges,v) || push!(forward_edges,v => Set{TypeVar}())
            push!(forward_edges[v],tv)
            push!(backward_edges[tv],v)
        end

    end
    Set{TypeVar}([tv])
end

_type_var_dependency(args...) = Set{TypeVar}()
_type_var_dependency(::Core.TypeofBottom,args...) = Set{TypeVar}()
_type_var_dependency(t::Union,args...) = begin
    mapreduce(_t->_type_var_dependency(_t,args...),union,collect_union(t);init=Set{TypeVar}())
end
_type_var_dependency(t::UnionAll,args...) = _type_var_dependency(Base.unwrap_unionall(t),args...)
_type_var_dependency(t::DataType,args...) = begin
    mapreduce(_t->_type_var_dependency(_t,args...),union,t.parameters;init=Set{TypeVar}())
end

"""
# Arguments
- `tvtable`: initial mapping of type variable changes, might contain stale entries 
- `fedges`: edges of dependency graph produced by [`type_var_dependency`](@ref)
- `bedges`: same as `fedges` but pointing backwards
"""
function propagate_all_type_var_changes!(tvtable,fedges,bedges)
    # topological sort on the type var graph using DFS
    tvs = TypeVar[]
    # we don't actually care about nodes not connected with any other nodes
    for v in keys(fedges)
        _topo_sort_visit!(v,fedges,tvs)
    end
    # we don't need to update nodes that
    # don't depend on other nodes
    filter!(v->!isempty(bedges[v]),tvs)
    for tv in tvs
        tv_new = update_type_var_ref(tv,tvtable)
        tvtable[tv] = tv_new
    end
    tvtable
end

function _topo_sort_visit!(vtx,edges,tvs)
    if vtx in tvs 
        nothing
    else
        if haskey(edges,vtx)
            for v in edges[vtx]
                _topo_sort_visit(v,edges,tvs)
            end
        end
        pushfirst!(tvs,vtx)
        nothing
    end
end

gpu_target_type(tv::TypeVar) = begin
    nt, tvr = gpu_target_type(tv.ub)
    (tv, _combine_tv_replacement(tvr, Dict(tv => TypeVar(tv.name,tv.lb,nt))))
end
gpu_target_type(::Type{AbstractArray}) = (GPUDestArray,Dict())
gpu_target_type(::Type{>:AbstractArray}) = (GPUDestArray,Dict())
gpu_target_type(t::Type{<:AbstractArray}) = begin
    (
        Union{filter(_t -> _t <: GPUDestArray,collect_union(Union{_unpack_abstract(t)}))...},
        Dict()
    )
end
# wrapper types
gpu_target_type(t::Type{<:GPUDestArray}) = begin
    reduce(((t1,tvr1), (t2,tvr2))-> (Union{t1,t2},_combine_tv_replacement(tvr1,tvr2)),
    map(_t -> begin
        if _t <: GPUArray
            (_t, Dict())
        else
            local n = parent_type_param_pos(_t)
            orig_t = _t
            _t = Base.unwrap_unionall(_t)
            old_param = _t.parameters[n]
            if old_param isa TypeVar
                replacement, tvr = gpu_target_type(old_param)
                (orig_t, tvr)
            else
                # not really possible since type parameters are always invariant in julia,
                # unless you put a type variable and specify its lower/upper bounds,
                # in which case the parameter itself shouldn't change here. The type variables
                # referenced by it will change, though.
                (Union{}, Dict())
            end
        end
    end, collect_union(flatten_union(_unpack_abstract(t)))); init = (Union{}, Dict()))
end
gpu_target_type(::Type{<:GPUArray}) = (Union{}, Dict())

SimpleTraits.trait(::Type{IsGPUOptimizable{X}}) where {X} = begin
    (_trecurse(X) <: GPUArray) ? IsGPUOptimizable{X} : Not{IsGPUOptimizable{X}}
end



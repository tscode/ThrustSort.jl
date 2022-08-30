module ThrustSort

using CUDA

const lib = joinpath(@__DIR__, "..", "deps", "thrust_sort.so")

const jtypes = [ Float32, Float64
               , Int8, Int16, Int32, Int64
               , UInt8, UInt16, UInt32, UInt64 ]

# Function body expression for calling sort functions of thrust
function gen_sort(:: Type{T}) where T
  fname = "thrust_sort_$T"
  sfname = "thrust_stable_sort_$T"
  quote
    function thrustsort!(val :: CuArray{$T}; stable = false)
      if !stable
        ccall(($fname, lib), Nothing, (CuPtr{$T}, UInt), val, length(val))
      else
        ccall(($sfname, lib), Nothing, (CuPtr{$T}, UInt), val, length(val))
      end
      val
    end
  end
end

# Function body expression for calling sort_by_key functions of thrust
function gen_sort_by_key(:: Type{T}, :: Type{S}) where {T, S}
  fname = "thrust_sort_by_key_$(T)_$(S)"
  sfname = "thrust_stable_sort_by_key_$(T)_$(S)"
  quote
    function thrustsort!(key :: CuArray{$T}, val :: CuArray{$S}; stable = false)
      @assert length(val) == length(key)
      if !stable
        ccall(($fname, lib), Nothing, (CuPtr{$T}, CuPtr{$S}, UInt), key, val, length(val))
      else
        ccall(($sfname, lib), Nothing, (CuPtr{$T}, CuPtr{$S}, UInt), key, val, length(val))
      end
      val
    end
  end
end

# Define sort functions 
for T in jtypes
  eval(gen_sort(T))
end

# Define sort_by_key functions 
for T in jtypes, S in jtypes
  eval(gen_sort_by_key(T, S))
end

thrustsort(val) = thrustsort!(copy(val))

export thrustsort!,
       thrustsort

end # module ThrustSort

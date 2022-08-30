
const jtypes = [ Float32, Float64
               , Int8, Int16, Int32, Int64
               , UInt8, UInt16, UInt32, UInt64 ]

const ctypes = [ "float", "double"
               , "char", "short", "int", "long long"
               , "unsigned char", "unsigned short", "unsigned int", "unsigned long long" ]

function codegen_header()
  """
  #include <thrust/device_ptr.h>
  #include <thrust/sort.h>

  """
end

function codegen_sort()
  mapreduce(vcat, enumerate(jtypes)) do (i, T)
    map(["", "stable_"]) do stable
      C = ctypes[i]
      """
      extern \"C\" void thrust_$(stable)sort_$T($C * value, size_t len) {
        thrust::device_ptr<$C> d_value(value);
        thrust::$(stable)sort(d_value, d_value + len);
      }
      """
    end
  end
end

function codegen_sort_by_key()
  mapreduce(vcat, enumerate(jtypes)) do (i, T)
    mapreduce(vcat, enumerate(jtypes)) do (j, S)
      map(["", "stable_"]) do stable
        C = ctypes[i]
        K = ctypes[j]
        """
        extern \"C\" void thrust_$(stable)sort_by_key_$(T)_$(S)($C * key, $K * value, size_t len) {
          thrust::device_ptr<$C> d_key(key);
          thrust::device_ptr<$K> d_value(value);
          thrust::$(stable)sort_by_key(d_key, d_key + len, d_value);
        }
        """
      end
    end
  end
end


# Create the .cu file

const cu_file = joinpath(@__DIR__, "thrust_sort.cu")
const so_file = joinpath(@__DIR__, "thrust_sort.so")

open(cu_file, "w") do file
  println(file, codegen_header())
  for entry in codegen_sort()
    println(file, entry)
  end
  for entry in codegen_sort_by_key()
    println(file, entry)
  end
end

# and try to compile it
try run(`which nvcc`)
  run(`nvcc --compiler-options '-fPIC' -o $so_file --shared $cu_file`)
catch _
  @warn "nvcc compiler not found. Package not built successfully"
end


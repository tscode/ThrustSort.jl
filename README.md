
# ThrustSort.jl
Wrapper of the sorting functionality of the cuda-powered thrust library, making it available to `CuArray`s.
It supports `CuArray`s with primitive element types `Float32`, `Float64`, `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`, `UInt32`, `UInt64`.
Sorting is currently restricted to the default comparison operator `<=`, but a second array with comparison keys can be provided.

## Installation

To install the package, run
```julia
using Pkg; Pkg.add(url="git@github.com:tscode/ThrustSort.jl.git")
```
The build process requires that the executable `nvcc` can be found (check via `which nvcc` in a shell before running julia),
and that the thrust header files are available. Building might take up to a couple of minutes.

## Usage

The interface is simple. The package exports the functions `thrustsort!` and its non-modifying variant `thrustsort`. Consider this example:
```julia
using CUDA
using ThrustSort

a = cu(rand(1000))
thrustsort!(a)
```
Depending on the size of the array, thrust's sorting can be orders of magnitude faster than the implementation available via the Base.sort interface.
Compare the following examples
```julia
a = cu(rand(100_0000_000))
CUDA.@time sort(a);
CUDA.@time thrustsort(a);
```
which returns timings of 19 versus 0.15 seconds on a Tesla M10 in the second run.

The function `thrustsort` also takes two arguments of type `CuArray`. In this case, the first argument is used as a key for the comparison.
Note that **both** arrays are sorted in this case.
```julia
key, val = cu([1, 3, 2]), cu([1, 2, 3])
thrustsort!(key, val) # switches element 2 and 3 in both a and b
```

### Sorting along dimensions
The primary purpose for this package was to accelerate sorting each column of
a `10000 x 10000` matrix. The in-built function `sort(a, dims = 1)` took several
seconds and converting the data to `CuArray` did not bring about performance
benefits.
Formally, column-wise sorting is not supported by `thrust::sort`. However, it
can be achieved via the following trick:
```julia
n = 10
a = cu(rand(n, n))
indices = cu(repeat(1:n, inner = n)) # indices[i] is the column of the value a[i]

# First, sort the data and apply the same permutation to indices
thrustsort!(a, indices)

# Now, sort indices without altering the order of elements with the same index 
thrustsort!(indices, a, stable = true)
```
This trick can be applied for arbitrary segmentations of the data. Even though
it looks inefficient (sorting `n^2` values twice), it improved runtimes from
more than 10 seconds to less than 1 second for `n = 10_000` (on the M10). In
fact, if we let `indices` be of type `Int16`, we can bring the runtime down to
about 0.3 seconds (without measuring the time to allocate `indices`, since it
can be reused). 

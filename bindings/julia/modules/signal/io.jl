# modules/signal/io.jl
# Signal I/O functions for binary file and SigMF format.

"""
    read_bin(filename::String, dtype_str::String, count::Int=-1) -> InsightArray

Read binary data from a file.

# Arguments
- `filename::String`: Path to the binary file.
- `dtype_str::String`: Data type string (e.g. `"float32"`, `"int16"`).
- `count::Int`: Number of elements to read. `-1` reads all.
"""
function read_bin end

"""
    write_bin(filename::String, x::InsightArray) -> Nothing

Write array data to a binary file.
"""
function write_bin end

"""
    pack_bin(x::InsightArray, bits::Int) -> InsightArray

Pack integer values into bit-packed representation.
"""
function pack_bin end

"""
    unpack_bin(x::InsightArray, bits::Int, count::Int) -> InsightArray

Unpack bit-packed data into integer values.
"""
function unpack_bin end

"""
    read_sigmf(filename::String, data_file::String="") -> InsightArray

Read data from a SigMF recording.
"""
function read_sigmf end

"""
    write_sigmf(filename::String, x::InsightArray, sample_rate::Float64,
                frequency::Float64=0.0) -> Nothing

Write data in SigMF format.
"""
function write_sigmf end

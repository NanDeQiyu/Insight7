// src/ops/manipulation.cpp
/**
 * @file manipulation.cpp
 * @brief Array manipulation operations.
 *
 * Provides shape manipulation, transposition, joining, splitting,
 * tiling, padding, rolling, and other array transformation operations.
 */

#include "insight/ops/manipulation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/creation.h"
#include "insight/core/op_registry.h"
#include "insight/core/exception.h"

namespace ins {

    static DeviceKind get_device_kind(const Place& place) {
        return place.is_cpu() ? DeviceKind::CPU : DeviceKind::GPU;
    }

    // ============================================================================
    // Helper: launch manipulation kernel
    // ============================================================================

    template<typename T>
    static void add_kernel_arg(std::vector<void*>& inputs, const T& arg) {
        if constexpr (std::is_pointer_v<std::decay_t<T>>) {
            inputs.push_back(const_cast<void*>(static_cast<const void*>(arg)));
        }
        else if constexpr (std::is_same_v<std::decay_t<T>, std::nullptr_t>) {
            inputs.push_back(nullptr);
        }
        else {
            inputs.push_back(const_cast<void*>(static_cast<const void*>(&arg)));
        }
    }

    /**
     * @brief Launch a manipulation kernel with common parameter pattern.
     *
     * @tparam Args Parameter pack types
     * @param kernel_name Kernel name to launch
     * @param x Input array
     * @param result Output array (pre-allocated)
     * @param extra_args Additional kernel arguments
     */
    template<typename... Args>
    static void launch_manipulation_kernel(const char* kernel_name,
        const Array& x,
        Array& result,
        Args&&... extra_args) {
        std::vector<void*> inputs;
        inputs.push_back(result.layout_ptr());
        inputs.push_back(const_cast<void*>(static_cast<const void*>(x.layout_ptr())));

        (add_kernel_arg(inputs, extra_args), ...);

        ops().launch(kernel_name, x.place(), x.dtype(),
            inputs,
            { result.layout_ptr() });
    }

    // ============================================================================
    // Shape manipulation (view operations, no kernel needed)
    // ============================================================================

    Array reshape(const Array& x, const Shape& new_shape) {
        return x.reshape(new_shape);
    }

    Array flatten(const Array& x) {
        return x.reshape(Shape({ x.numel() }));
    }

    Array ravel(const Array& x) {
        if (x.is_contiguous()) {
            return flatten(x);
        }
        else {
            return x.copy().reshape(Shape({ x.numel() }));
        }
    }

    Array squeeze(const Array& x) {
        return x.squeeze();
    }

    Array squeeze(const Array& x, int axis) {
        return x.squeeze(axis);
    }

    Array unsqueeze(const Array& x, int axis) {
        return x.unsqueeze(axis);
    }

    // ============================================================================
    // Transpose (view operations, no kernel needed)
    // ============================================================================

    Array transpose(const Array& x) {
        return x.transpose();
    }

    Array permute(const Array& x, const std::vector<int>& axes) {
        return x.transpose(axes);
    }

    Array swapaxes(const Array& x, int axis1, int axis2) {
        Shape shape = x.shape();
        int ndim = shape.ndim();
        if (axis1 < 0) axis1 += ndim;
        if (axis2 < 0) axis2 += ndim;
        INS_CHECK(axis1 >= 0 && axis1 < ndim, "swapaxes: axis1 out of range");
        INS_CHECK(axis2 >= 0 && axis2 < ndim, "swapaxes: axis2 out of range");

        std::vector<int> perm(ndim);
        for (int i = 0; i < ndim; ++i) perm[i] = i;
        std::swap(perm[axis1], perm[axis2]);
        return x.transpose(perm);
    }

    Array moveaxis(const Array& x, int source, int destination) {
        Shape shape = x.shape();
        int ndim = shape.ndim();
        if (source < 0) source += ndim;
        if (destination < 0) destination += ndim;
        INS_CHECK(source >= 0 && source < ndim, "moveaxis: source out of range");
        INS_CHECK(destination >= 0 && destination < ndim, "moveaxis: destination out of range");

        if (source == destination) return x;

        std::vector<int> perm;
        for (int i = 0; i < ndim; ++i) {
            if (i != source) perm.push_back(i);
        }
        perm.insert(perm.begin() + destination, source);
        return x.transpose(perm);
    }

    // ============================================================================
    // Flipping and rotating
    // ============================================================================

    Array flip(const Array& x, std::optional<int> axis) {
        if (!axis.has_value()) {
            // Flip all axes
            Array result = x;
            for (int i = 0; i < x.shape().ndim(); ++i) {
                result = flip(result, i);
            }
            return result;
        }

        int ax = axis.value();
        int ndim = x.shape().ndim();
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "flip: axis out of range");

        Array result(x.shape(), x.dtype(), x.place());

        ops().launch("flip", x.place(), x.dtype(),
            { (void*)result.layout_ptr(), (void*)x.layout_ptr(), &ax },
            { (void*)result.layout_ptr() });

        return result;
    }

    Array rot90(const Array& x, int k, const std::vector<int>& axes) {
        INS_CHECK(x.shape().ndim() >= 2, "rot90: requires at least 2 dimensions");
        INS_CHECK(axes.size() == 2, "rot90: axes must have exactly 2 elements");

        int ndim = x.shape().ndim();
        int axis1 = axes[0];
        int axis2 = axes[1];
        if (axis1 < 0) axis1 += ndim;
        if (axis2 < 0) axis2 += ndim;
        INS_CHECK(axis1 >= 0 && axis1 < ndim, "rot90: axis1 out of range");
        INS_CHECK(axis2 >= 0 && axis2 < ndim, "rot90: axis2 out of range");
        INS_CHECK(axis1 != axis2, "rot90: axes must be different");

        // Normalize k to [0, 3]
        int k_mod = ((k % 4) + 4) % 4;
        if (k_mod == 0) return x.copy();

        // Build permutation that swaps axis1 and axis2
        std::vector<int> perm(ndim);
        for (int i = 0; i < ndim; ++i) perm[i] = i;
        perm[axis1] = axis2;
        perm[axis2] = axis1;

        Array result = x;
        for (int i = 0; i < k_mod; ++i) {
            // One 90-degree counter-clockwise rotation: transpose then flip
            result = result.transpose(perm);
            result = flip(result, axis1);  // axis1 is the new position of axis2
        }

        return result;
    }

    // ============================================================================
    // Joining
    // ============================================================================

    Array concat(const std::vector<Array>& tensors, int axis) {
        if (tensors.empty()) INS_THROW("concat: no tensors provided");
        if (tensors.size() == 1) return tensors[0];

        const Shape& first_shape = tensors[0].shape();
        int ndim = first_shape.ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "concat: axis out of range");

        // Check compatibility and compute output shape
        DType dtype = tensors[0].dtype();
        Place place = tensors[0].place();
        std::vector<int64_t> out_dims = first_shape.dims();
        int64_t concat_size = 0;

        for (const auto& t : tensors) {
            INS_CHECK(t.dtype() == dtype, "concat: dtype mismatch");
            INS_CHECK(t.place() == place, "concat: device mismatch");
            INS_CHECK(t.shape().ndim() == ndim, "concat: dimension mismatch");
            for (int i = 0; i < ndim; ++i) {
                if (i != ax) {
                    INS_CHECK(t.shape().dim(i) == out_dims[i],
                        "concat: shape mismatch at dimension ", i);
                }
            }
            concat_size += t.shape().dim(ax);
        }
        out_dims[ax] = concat_size;
        Shape out_shape(out_dims);

        Array result(out_shape, dtype, place);

        // Prepare inputs: [output, num_inputs, tensor0, tensor1, ..., tensorN, axis]
        std::vector<void*> inputs;
        inputs.push_back(result.layout_ptr());           // output (unused, kernel uses outputs[0])

        int num_tensors = static_cast<int>(tensors.size());
        inputs.push_back(const_cast<void*>(static_cast<const void*>(&num_tensors)));

        for (const auto& t : tensors) {
            inputs.push_back(const_cast<void*>(static_cast<const void*>(t.layout_ptr())));
        }

        inputs.push_back(const_cast<void*>(static_cast<const void*>(&ax)));

        ops().launch("concat", place, dtype, inputs, { result.layout_ptr() });

        return result;
    }

    Array stack(const std::vector<Array>& tensors, int axis) {
        if (tensors.empty()) INS_THROW("stack: no tensors provided");

        const Shape& first_shape = tensors[0].shape();
        int ndim = first_shape.ndim();
        int ax = axis;
        if (ax < 0) ax += ndim + 1;
        INS_CHECK(ax >= 0 && ax <= ndim, "stack: axis out of range");

        // Check all tensors have same shape and dtype
        DType dtype = tensors[0].dtype();
        Place place = tensors[0].place();
        for (const auto& t : tensors) {
            INS_CHECK(t.shape() == first_shape, "stack: shape mismatch");
            INS_CHECK(t.dtype() == dtype, "stack: dtype mismatch");
            INS_CHECK(t.place() == place, "stack: device mismatch");
        }

        // First unsqueeze each tensor, then concat
        std::vector<Array> expanded;
        expanded.reserve(tensors.size());
        for (const auto& t : tensors) {
            expanded.push_back(unsqueeze(t, ax));
        }
        return concat(expanded, ax);
    }

    // ============================================================================
    // Splitting (view operations, no kernel needed)
    // ============================================================================

    std::vector<Array> split(const Array& x, int indices_or_sections, int axis) {
        Shape shape = x.shape();
        int ndim = shape.ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "split: axis out of range");

        int64_t dim_size = shape.dim(ax);
        if (dim_size % indices_or_sections != 0) {
            INS_THROW("split: axis dimension must be divisible by number of splits");
        }

        int64_t split_size = dim_size / indices_or_sections;
        std::vector<int64_t> indices;
        for (int64_t i = split_size; i < dim_size; i += split_size) {
            indices.push_back(i);
        }
        return split(x, indices, ax);
    }

    std::vector<Array> split(const Array& x, const std::vector<int64_t>& indices, int axis) {
        Shape shape = x.shape();
        int ndim = shape.ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "split: axis out of range");

        std::vector<Array> result;
        int64_t start = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            int64_t end = indices[i];
            std::vector<Slice> slices(ndim, Slice::all());
            slices[ax] = Slice(start, end);
            result.push_back(x.slice(slices));
            start = end;
        }

        std::vector<Slice> slices(ndim, Slice::all());
        slices[ax] = Slice(start, shape.dim(ax));
        result.push_back(x.slice(slices));

        return result;
    }

    // ============================================================================
    // Tiling and Repeating
    // ============================================================================

    Array repeat(const Array& x, int repeats, std::optional<int> axis) {
        if (!axis.has_value()) {
            Array flat = ravel(x);
            return repeat(flat, repeats, 0);
        }

        int ax = axis.value();
        int ndim = x.shape().ndim();
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "repeat: axis out of range");
        INS_CHECK(repeats >= 0, "repeat: repeats must be non-negative");

        // Compute output shape
        std::vector<int64_t> out_dims = x.shape().dims();
        out_dims[ax] *= repeats;
        Shape out_shape(out_dims);

        Array result(out_shape, x.dtype(), x.place());

        std::vector<void*> inputs;
        inputs.push_back(const_cast<void*>(static_cast<const void*>(x.layout_ptr())));
        inputs.push_back(const_cast<void*>(static_cast<const void*>(&repeats)));
        inputs.push_back(const_cast<void*>(static_cast<const void*>(&ax)));

        ops().launch("repeat", x.place(), x.dtype(),
            { (void*)result.layout_ptr(), (void*)x.layout_ptr(), &repeats, &ax },
            { (void*)result.layout_ptr() });

        return result;
    }

    Array tile(const Array& x, const Shape& reps) {
        // Compute output shape
        Shape in_shape = x.shape();
        int in_ndim = in_shape.ndim();
        int out_ndim = std::max(in_ndim, reps.ndim());

        std::vector<int64_t> out_dims(out_ndim, 1);
        for (int i = 0; i < out_ndim; ++i) {
            int in_idx = i - (out_ndim - in_ndim);
            int64_t in_dim = (in_idx >= 0) ? in_shape.dim(in_idx) : 1;
            int64_t rep = (i < reps.ndim()) ? reps.dim(i) : 1;
            out_dims[i] = in_dim * rep;
        }
        Shape out_shape(out_dims);

        Array result(out_shape, x.dtype(), x.place());

        launch_manipulation_kernel("tile", x, result, reps);

        return result;
    }

    // ============================================================================
    // Padding
    // ============================================================================

    Array pad(const Array& x, const std::vector<int64_t>& pad_width, double constant_value) {
        Shape in_shape = x.shape();
        int ndim = in_shape.ndim();
        INS_CHECK(pad_width.size() == static_cast<size_t>(2 * ndim),
            "pad: pad_width size mismatch");

        // Compute output shape
        std::vector<int64_t> out_dims(ndim);
        for (int i = 0; i < ndim; ++i) {
            out_dims[i] = in_shape.dim(i) + pad_width[2 * i] + pad_width[2 * i + 1];
        }
        Shape out_shape(out_dims);

        Array result(out_shape, x.dtype(), x.place());

        int64_t* pad_width_ptr = const_cast<int64_t*>(pad_width.data());
        launch_manipulation_kernel("pad", x, result, pad_width_ptr, constant_value);

        return result;
    }

    // ============================================================================
    // Rolling
    // ============================================================================

    Array roll(const Array& x, int shift, std::optional<int> axis) {
        int ax = axis.has_value() ? axis.value() : -1;
        Array result(x.shape(), x.dtype(), x.place());

        launch_manipulation_kernel("roll", x, result, shift, ax);

        return result;
    }

    // ============================================================================
    // Diagonal
    // ============================================================================

    Array diag(const Array& x, int k) {
        const Shape& shape = x.shape();
        Array result;

        if (shape.ndim() == 1) {
            // Construct diagonal matrix from 1D array
            int64_t n = x.numel();
            int64_t size = n + std::abs(k);
            Shape out_shape({ size, size });
            result = Array(out_shape, x.dtype(), x.place());
        }
        else if (shape.ndim() == 2) {
            // Extract diagonal from 2D array
            int64_t rows = shape.dim(0);
            int64_t cols = shape.dim(1);
            int64_t diag_len;
            if (k >= 0) {
                diag_len = std::min(rows, cols - k);
            }
            else {
                diag_len = std::min(rows + k, cols);
            }
            Shape out_shape({ diag_len });
            result = Array(out_shape, x.dtype(), x.place());
        }
        else {
            INS_THROW("diag: input must be 1D or 2D");
        }

        launch_manipulation_kernel("diag", x, result, k);

        return result;
    }

    Array diagonal(const Array& x, int offset, int axis1, int axis2) {
        // For 2D arrays, this is similar to diag
        return diag(x, offset);
    }

    // ============================================================================
    // Triangular
    // ============================================================================

    Array tril(const Array& x, int k) {
        Array result(x.shape(), x.dtype(), x.place());
        launch_manipulation_kernel("tril", x, result, k);
        return result;
    }

    Array triu(const Array& x, int k) {
        Array result(x.shape(), x.dtype(), x.place());
        launch_manipulation_kernel("triu", x, result, k);
        return result;
    }

    // ============================================================================
    // Slicing (Views) - Direct Array methods
    // ============================================================================

    Array slice(Array& x, int dim, int64_t start, int64_t stop, int64_t step) {
        return x.slice(dim, start, stop, step);
    }

    Array slice(Array& x, const std::vector<Slice>& slices) {
        return x.slice(slices);
    }

    // ============================================================================
    // diff
    // ============================================================================

    Array diff(const Array& x, int n, int axis) {
        INS_CHECK(x.defined(), "diff: input is undefined");
        INS_CHECK(n >= 0, "diff: n must be non-negative");

        if (n == 0) {
            return x.copy();
        }

        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "diff: axis out of range");

        Array result = x;
        for (int i = 0; i < n; ++i) {
            int64_t axis_size = result.shape().dim(ax);
            INS_CHECK(axis_size > 1, "diff: axis size must be at least 2");

            std::vector<Slice> slices_front(ndim, Slice::all());
            std::vector<Slice> slices_back(ndim, Slice::all());
            slices_front[ax] = Slice(0, axis_size - 1);
            slices_back[ax] = Slice(1, axis_size);

            Array front = result.slice(slices_front);
            Array back = result.slice(slices_back);

            result = sub(back, front);
        }

        return result;
    }

    // ============================================================================
    // contiguous
    // ============================================================================

    Array contiguous(const Array& x) {
        if (x.is_contiguous()) return x;

        Array result(x.shape(), x.dtype(), x.place());

        ops().launch("contiguous_copy", x.place(), x.dtype(),
            { (void*)x.layout_ptr() },
            { result.layout_ptr() });

        return result;
    }

} // namespace ins
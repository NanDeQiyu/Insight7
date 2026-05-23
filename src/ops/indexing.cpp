// src/ops/indexing.cpp
/**
 * @file indexing.cpp
 * @brief Indexing operations for array manipulation.
 *
 * Provides advanced indexing operations including take, put, where,
 * nonzero, argsort, topk, unique, and other array indexing utilities.
 */

#include "insight/ops/indexing.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/broadcast.h"
#include "insight/ops/creation.h"
#include "insight/ops/reduction.h"
#include "insight/ops/manipulation.h"
#include "insight/core/op_registry.h"
#include "insight/utils/promotion.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <insight/io/print.h>
namespace ins {

    static DeviceKind get_device_kind(const Place& place) {
        return place.is_cpu() ? DeviceKind::CPU : DeviceKind::GPU;
    }

    // ============================================================================
    // Helper: Prepare flattened array for indexing
    // ============================================================================

    /**
     * @brief Prepare array for indexing by flattening or making contiguous.
     *
     * @param x Input array
     * @param axis Optional axis (if provided, only make contiguous)
     * @return Prepared array
     */
    static Array prepare_flattened(const Array& x, std::optional<int> axis) {
        if (!axis.has_value()) {
            return x.reshape(Shape({ x.numel() }));
        }
        return x.contiguous();
    }

    /**
     * @brief Broadcast shapes for take_along_axis operation.
     *
     * @param x_shape Shape of the input array
     * @param idx_shape Shape of the indices array
     * @param axis The axis along which to take
     * @return Broadcasted output shape
     */
    static Shape broadcast_shapes_for_indexing(const Shape& x_shape,
        const Shape& idx_shape,
        int axis) {
        int ndim = x_shape.ndim();
        std::vector<int64_t> out_dims(ndim);
        for (int i = 0; i < ndim; ++i) {
            if (i == axis) {
                out_dims[i] = idx_shape.dim(i);
            }
            else {
                int64_t xdim = x_shape.dim(i);
                int64_t idxdim = (i < idx_shape.ndim()) ? idx_shape.dim(i) : 1;
                if (xdim != idxdim && xdim != 1 && idxdim != 1) {
                    INS_THROW("broadcast_shapes_for_indexing: shape mismatch at axis ", i,
                        ": ", xdim, " vs ", idxdim);
                }
                out_dims[i] = std::max(xdim, idxdim);
            }
        }
        return Shape(out_dims);
    }

    // ============================================================================
    // take
    // ============================================================================

    /**
     * @brief Take elements from an array along an axis.
     *
     * @param x Input array
     * @param indices Indices to take (1D array)
     * @param axis Axis along which to take (optional, if not provided, flatten)
     * @return Array of taken elements
     */
    Array take(const Array& x, const Array& indices, std::optional<int> axis) {
        Array prepared = prepare_flattened(x, axis);

        Array idx = indices;
        if (idx.dtype() != DType::I64) {
            idx = idx.to(DType::I64);
        }
        if (idx.place() != x.place()) {
            idx = idx.to(x.place());
        }

        Shape out_shape;
        if (axis.has_value()) {
            std::vector<int64_t> dims = x.shape().dims();
            int ax = axis.value();
            if (ax < 0) ax += x.shape().ndim();
            dims[ax] = idx.numel();
            out_shape = Shape(dims);
        }
        else {
            out_shape = Shape({ idx.numel() });
        }

        Array result(out_shape, x.dtype(), x.place());

        int normalized_axis = axis.value_or(-1);
        bool has_axis = axis.has_value();

        ops().launch("take", x.place(), x.dtype(),
            { (void*)result.layout_ptr(), prepared.layout_ptr(), (void*)idx.layout_ptr(),
              &normalized_axis, &has_axis },
            { (void*)result.layout_ptr() });

        return result;
    }

    // ============================================================================
    // take_along_axis
    // ============================================================================

    /**
     * @brief Take values from the input array using indices array along an axis.
     *
     * @param x Input array
     * @param indices Indices array (same shape as output except on axis)
     * @param axis Axis along which to take
     * @return Array of taken values
     */
    Array take_along_axis(const Array& x, const Array& indices, int axis) {
        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "take_along_axis: axis out of range");

        Array idx = indices;
        if (idx.dtype() != DType::I64) {
            idx = idx.to(DType::I64);
        }
        if (idx.place() != x.place()) {
            idx = idx.to(x.place());
        }

        Shape out_shape = broadcast_shapes_for_indexing(x.shape(), idx.shape(), ax);

        Array idx_broadcasted = broadcast_to(idx, out_shape);

        Array result(out_shape, x.dtype(), x.place());

        ops().launch("take_along_axis", x.place(), x.dtype(),
            { (void*)result.layout_ptr(), (void*)x.layout_ptr(), (void*)idx_broadcasted.layout_ptr(), &ax },
            { (void*)result.layout_ptr() });

        return result;
    }

    // ============================================================================
    // put
    // ============================================================================

    /**
     * @brief Put values into an array at specified indices.
     *
     * @param x Input array (will be copied)
     * @param indices Indices where to put values
     * @param values Values to put
     * @param axis Axis along which to put (optional)
     * @return Array with values placed
     */
    Array put(const Array& x, const Array& indices, const Array& values, std::optional<int> axis) {
        Array prepared = prepare_flattened(x, axis);

        Array idx = indices;
        if (idx.dtype() != DType::I64) {
            idx = idx.to(DType::I64);
        }
        if (idx.place() != x.place()) {
            idx = idx.to(x.place());
        }

        Array val = values;
        if (val.numel() == 1) {
            val = broadcast_to(val, Shape({ idx.numel() }));
        }
        INS_CHECK(val.numel() == idx.numel(), "put: values must broadcast to indices shape");

        Array result = prepared.copy();

        ops().launch("put", x.place(), x.dtype(),
            { (void*)result.layout_ptr(), (void*)idx.layout_ptr(), val.layout_ptr() },
            { (void*)result.layout_ptr() });

        if (!axis.has_value()) {
            return result.reshape(x.shape());
        }
        return result;
    }

    // ============================================================================
    // put_along_axis
    // ============================================================================

    /**
     * @brief Put values into an array along an axis using indices.
     *
     * @param x Input array (will be copied)
     * @param indices Indices where to put values
     * @param values Values to put
     * @param axis Axis along which to put
     * @return Array with values placed
     */
    Array put_along_axis(const Array& x, const Array& indices, const Array& values, int axis) {
        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "put_along_axis: axis out of range");

        Array idx = indices;
        if (idx.dtype() != DType::I64) {
            idx = idx.to(DType::I64);
        }
        if (idx.place() != x.place()) {
            idx = idx.to(x.place());
        }

        Array val = values;
        if (val.dtype() != x.dtype()) {
            val = val.to(x.dtype());
        }
        if (val.place() != x.place()) {
            val = val.to(x.place());
        }

        if (val.numel() == 1) {
            Shape target_shape = broadcast_shapes_for_indexing(x.shape(), idx.shape(), ax);
            val = broadcast_to(val, target_shape);
        }

        Array result = x.copy();

        ops().launch("put_along_axis", x.place(), x.dtype(),
            { (void*)result.layout_ptr(), (void*)idx.layout_ptr(), (void*)val.layout_ptr(), (void*)&ax },
            { (void*)result.layout_ptr() });

        return result;
    }

    // ============================================================================
    // gather / scatter (aliases)
    // ============================================================================

    /**
     * @brief Gather values along an axis using indices (alias for take_along_axis).
     */
    Array gather(const Array& x, int dim, const Array& index) {
        return take_along_axis(x, index, dim);
    }

    /**
     * @brief Scatter values into an array (alias for scatter_reduce with replace mode).
     */
    Array scatter(const Array& x, int dim, const Array& index, const Array& src) {
        return scatter_reduce(x, dim, index, src, "replace");
    }

    /**
     * @brief Scatter and add values into an array.
     */
    Array scatter_add(const Array& x, int dim, const Array& index, const Array& src) {
        return scatter_reduce(x, dim, index, src, "add");
    }

    // ============================================================================
    // scatter_reduce
    // ============================================================================

    /**
     * @brief Scatter values into an array with reduction.
     *
     * @param x Input array (will be copied)
     * @param dim Dimension along which to scatter
     * @param index Indices where to scatter
     * @param src Source values
     * @param reduce Reduction mode: "replace", "add", "mul", "max", "min"
     * @return Array with scattered values
     */
    Array scatter_reduce(const Array& x, int dim, const Array& index, const Array& src,
        const std::string& reduce) {
        int ndim = x.shape().ndim();
        int d = dim;
        if (d < 0) d += ndim;
        INS_CHECK(d >= 0 && d < ndim, "scatter_reduce: dim out of range");

        Array idx = index;
        if (idx.dtype() != DType::I64) {
            idx = idx.to(DType::I64);
        }
        if (idx.place() != x.place()) {
            idx = idx.to(x.place());
        }

        Array src_broadcast = src;
        if (src_broadcast.shape() != idx.shape()) {
            src_broadcast = broadcast_to(src_broadcast, idx.shape());
        }

        Array result = x.copy();

        ops().launch("scatter_reduce", x.place(), x.dtype(),
            { (void*)result.layout_ptr(), (void*)idx.layout_ptr(), (void*)src_broadcast.layout_ptr(), (void*)&d, (void*)reduce.c_str()},
            { (void*)result.layout_ptr() });

        return result;
    }

    // ============================================================================
    // masked_select
    // ============================================================================

    /**
     * @brief Select elements from an array where mask is true.
     *
     * @param x Input array
     * @param mask Boolean mask array
     * @return 1D array of selected elements
     */
    Array masked_select(const Array& x, const Array& mask) {
        Array condition = mask;
        if (condition.dtype() != DType::BOOL) {
            condition = condition.to(DType::BOOL);
        }
        if (condition.place() != x.place()) {
            condition = condition.to(x.place());
        }

        if (condition.shape() != x.shape()) {
            condition = broadcast_to(condition, x.shape());
        }

        // First pass: count number of true elements
        Array flattened_cond = condition.reshape(Shape({ condition.numel() }));
        Array cnt = count_nonzero(flattened_cond);
        int64_t count = cnt.item<int64_t>();

        Array result(Shape({ count }), x.dtype(), x.place());

        ops().launch("masked_select", x.place(), x.dtype(),
            { (void*)result.layout_ptr(), (void*)x.layout_ptr(), condition.layout_ptr() },
            { (void*)result.layout_ptr() });

        return result;
    }

    // ============================================================================
    // compress
    // ============================================================================

    /**
     * @brief Select slices along an axis where condition is true.
     *
     * @param x Input array
     * @param condition 1D boolean array
     * @param axis Axis along which to compress
     * @return Compressed array
     */
    Array compress(const Array& x, const Array& condition, std::optional<int> axis) {
        int ndim = x.shape().ndim();
        int ax = axis.value_or(0);
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "compress: axis out of range");

        Array cond = condition;
        if (cond.dtype() != DType::BOOL) {
            cond = cond.to(DType::BOOL);
        }

        int64_t axis_dim = x.shape().dim(ax);
        if (cond.numel() != axis_dim) {
            INS_THROW("compress: condition length must match axis dimension");
        }

        Array cnt = count_nonzero(cond);
        int64_t keep_count = cnt.item<int64_t>();

        std::vector<int64_t> out_dims = x.shape().dims();
        out_dims[ax] = keep_count;
        Shape out_shape(out_dims);

        Array result(out_shape, x.dtype(), x.place());

        ops().launch("compress", x.place(), x.dtype(),
            { (void*)result.layout_ptr(), (void*)x.layout_ptr(), cond.layout_ptr(), &ax },
            { (void*)result.layout_ptr() });

        return result;
    }

    // ============================================================================
    // where
    // ============================================================================

    /**
     * @brief Return elements chosen from x or y depending on condition.
     *
     * @param condition Boolean condition array
     * @param x Values to take where condition is true
     * @param y Values to take where condition is false
     * @return Array of same shape as condition
     */
    Array where(const Array& condition, const Array& x, const Array& y) {
        auto broadcasted = broadcast_arrays({ condition, x, y });
        Array cond = broadcasted[0];
        if (cond.dtype() != DType::BOOL) {
            cond = cond.to(DType::BOOL);
        }
        Array X = broadcasted[1];
        Array Y = broadcasted[2];

        Array result(cond.shape(), X.dtype(), X.place());

        ops().launch("where", condition.place(), X.dtype(),
            { (void*)result.layout_ptr(), cond.layout_ptr(), X.layout_ptr(), Y.layout_ptr() },
            { (void*)result.layout_ptr() });

        return result;
    }

    // ============================================================================
    // nonzero
    // ============================================================================

    /**
     * @brief Return the indices of the non-zero elements.
     *
     * @param x Input array
     * @return 2D array of shape (ndim, nz_count) containing indices
     */
    Array nonzero(const Array& x) {
        // Create placeholder output array (will be filled by kernel)
        Array result;

        ops().launch("nonzero", x.place(), x.dtype(),
            { (void*)(void*)x.layout_ptr(), (void*)(void*)result.layout_ptr() },
            { (void*)(void*)result.layout_ptr() });

        return Array(result.layout_ptr());
    }

    // ============================================================================
    // flatnonzero
    // ============================================================================

    /**
     * @brief Return the indices of the non-zero elements in the flattened array.
     *
     * @param x Input array
     * @return 1D array of indices
     */
    Array flatnonzero(const Array& x) {
        Array flat_x = x.reshape(Shape({ x.numel() }));
        Array nz = nonzero(flat_x);

        if (nz.numel() == 0) {
            return nz;
        }
        // nz shape is (1, n) for 1D input, flatten to (n,)
        return nz.reshape(Shape({ nz.numel() }));
    }

    // ============================================================================
    // argsort
    // ============================================================================

    /**
     * @brief Return the indices that would sort the array.
     *
     * @param x Input array
     * @param axis Axis along which to sort
     * @param descending Whether to sort in descending order
     * @return Indices array of same shape as x
     */
    Array argsort(const Array& x, int axis, bool descending) {
        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "argsort: axis out of range");

        // Move axis to last dimension for contiguous processing
        Array prepared = x;
        std::vector<int> perm(ndim);
        for (int i = 0; i < ndim; ++i) perm[i] = i;

        if (ax != ndim - 1) {
            std::swap(perm[ax], perm[ndim - 1]);
            prepared = prepared.transpose(perm);
            prepared = prepared.contiguous();
        }

        Shape out_shape = x.shape();
        Array result(out_shape, DType::I64, x.place());

        ops().launch("argsort", x.place(), x.dtype(),
            { (void*)result.layout_ptr(), prepared.layout_ptr(), &descending },
            { (void*)result.layout_ptr() });

        // Transpose back if needed
        if (ax != ndim - 1) {
            std::vector<int> inv_perm(ndim);
            for (int i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;
            result = result.transpose(inv_perm);
        }

        return result;
    }

    // ============================================================================
    // sort
    // ============================================================================

    /**
     * @brief Return a sorted copy of the array.
     *
     * @param x Input array
     * @param axis Axis along which to sort
     * @param descending Whether to sort in descending order
     * @return Sorted array
     */
    Array sort(const Array& x, int axis, bool descending) {
        Array indices = argsort(x, axis, descending);
        return take_along_axis(x, indices, axis);
    }

    // ============================================================================
    // topk
    // ============================================================================

    /**
     * @brief Return the top k values and their indices.
     *
     * @param x Input array
     * @param k Number of top elements to return
     * @param axis Axis along which to find top elements
     * @param largest Whether to return largest (true) or smallest (false)
     * @param sorted Whether to return sorted results
     * @return Tuple of (values, indices)
     */
    std::tuple<Array, Array> topk(const Array& x, int64_t k, int axis,
        bool largest, bool sorted) {
        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "topk: axis out of range");
        INS_CHECK(k > 0, "topk: k must be positive");

        int64_t axis_size = x.shape().dim(ax);
        if (k > axis_size) k = axis_size;

        // Move axis to last dimension for contiguous processing
        Array prepared = x;
        std::vector<int> perm(ndim);
        for (int i = 0; i < ndim; ++i) perm[i] = i;

        if (ax != ndim - 1) {
            std::swap(perm[ax], perm[ndim - 1]);
            prepared = prepared.transpose(perm);
            prepared = prepared.contiguous();
        }

        std::vector<int64_t> out_dims = prepared.shape().dims();
        out_dims.back() = k;
        Shape out_shape(out_dims);

        Array values(out_shape, x.dtype(), x.place());
        Array indices(out_shape, DType::I64, x.place());

        ops().launch("topk", x.place(), x.dtype(),
            { values.layout_ptr(), indices.layout_ptr(), prepared.layout_ptr(),
              &k, &largest, &sorted },
            { values.layout_ptr(), indices.layout_ptr() });

        // Transpose back if needed
        if (ax != ndim - 1) {
            std::vector<int> inv_perm(ndim);
            for (int i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;
            values = values.transpose(inv_perm);
            indices = indices.transpose(inv_perm);
        }

        return { values, indices };
    }

    // ============================================================================
    // searchsorted
    // ============================================================================

    /**
     * @brief Find indices where elements should be inserted to maintain order.
     *
     * @param x Sorted 1D array
     * @param v Values to insert
     * @param side 'left' or 'right'
     * @param sorter Optional indices that sort x
     * @return Indices array of same shape as v
     */
    Array searchsorted(const Array& x, const Array& v, const std::string& side,
        std::optional<Array> sorter) {
        INS_CHECK(x.shape().ndim() == 1, "searchsorted: x must be 1D");

        Array sorted_x = x;
        if (sorter.has_value()) {
            sorted_x = take(sorted_x, sorter.value());
        }

        Array result(v.shape(), DType::I64, v.place());

        int side_code = (side == "left") ? 0 : 1;

        ops().launch("searchsorted", x.place(), x.dtype(),
            { (void*)result.layout_ptr(), (void*)sorted_x.layout_ptr(), (void*)v.layout_ptr(), (void*)&side_code },
            { (void*)result.layout_ptr() });

        return result;
    }

    // ============================================================================
    // unique
    // ============================================================================

    /**
     * @brief Return the unique elements of an array.
     *
     * @param x Input array
     * @param return_indices Whether to return indices of first occurrences
     * @param return_inverse Whether to return inverse indices
     * @param return_counts Whether to return counts
     * @return UniqueResult structure containing requested arrays
     */
    UniqueResult unique(const Array& x, bool return_indices,
        bool return_inverse, bool return_counts) {
        Array flattened = x.reshape(Shape({ x.numel() }));

        // Create placeholder outputs (will be filled by kernel)
        Array unique_arr(Shape({ 0 }), x.dtype(), x.place());
        Array indices_arr;
        Array inverse_arr;
        Array counts_arr;

        std::vector<Array> outputs;
        outputs.push_back(unique_arr);

        if (return_indices) {
            indices_arr = Array(Shape({ 0 }), DType::I64, x.place());
            outputs.push_back(indices_arr);
        }
        if (return_inverse) {
            inverse_arr = Array(Shape({ 0 }), DType::I64, x.place());
            outputs.push_back(inverse_arr);
        }
        if (return_counts) {
            counts_arr = Array(Shape({ 0 }), DType::I64, x.place());
            outputs.push_back(counts_arr);
        }

        // Prepare inputs
        std::vector<void*> inputs;
        inputs.push_back(flattened.layout_ptr());
        inputs.push_back(const_cast<void*>(static_cast<const void*>(&return_indices)));
        inputs.push_back(const_cast<void*>(static_cast<const void*>(&return_inverse)));
        inputs.push_back(const_cast<void*>(static_cast<const void*>(&return_counts)));

        std::vector<void*> output_ptrs;
        for (auto& out : outputs) {
            output_ptrs.push_back(out.layout_ptr());
        }

        ops().launch("unique", x.place(), x.dtype(), inputs, output_ptrs);

        UniqueResult result;
        result.unique = outputs[0];
        size_t idx = 1;
        if (return_indices) {
            result.indices = outputs[idx++];
        }
        if (return_inverse) {
            result.inverse = outputs[idx++];
        }
        if (return_counts) {
            result.counts = outputs[idx++];
        }
        return result;
    }

    // ============================================================================
    // partition
    // ============================================================================

    /**
     * @brief Return a partially sorted copy of the array.
     *
     * @param x Input array
     * @param kth Element index that will be in its sorted position
     * @param axis Axis along which to partition
     * @return Partially sorted array
     */
    Array partition(const Array& x, int64_t kth, int axis) {
        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "partition: axis out of range");
        INS_CHECK(kth >= 0 && kth < x.shape().dim(ax),
            "partition: kth out of range");

        Array result(x.shape(), x.dtype(), x.place());

        ops().launch("partition", x.place(), x.dtype(),
            { (void*)result.layout_ptr(), (void*)x.layout_ptr(), &kth, &ax },
            { (void*)result.layout_ptr() });

        return result;
    }

    // ============================================================================
    // argpartition
    // ============================================================================

    /**
     * @brief Return the indices that would partially sort the array.
     *
     * @param x Input array
     * @param kth Element index that will be in its sorted position
     * @param axis Axis along which to partition
     * @return Indices array
     */
    Array argpartition(const Array& x, int64_t kth, int axis) {
        int ndim = x.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ax >= 0 && ax < ndim, "argpartition: axis out of range");
        INS_CHECK(kth >= 0 && kth < x.shape().dim(ax),
            "argpartition: kth out of range");

        Array result(x.shape(), DType::I64, x.place());

        ops().launch("argpartition", x.place(), x.dtype(),
            { (void*)result.layout_ptr(), (void*)x.layout_ptr(), &kth, &ax },
            { (void*)result.layout_ptr() });

        return result;
    }

    // ============================================================================
    // lexsort
    // ============================================================================

    /**
     * @brief Perform an indirect stable sort using a sequence of keys.
     *
     * @param keys Array of keys (first dimension is number of keys)
     * @param axis Axis along which to sort (default -1)
     * @return Indices that sort the keys
     */
    Array lexsort(const Array& keys, int axis) {
        int ndim = keys.shape().ndim();
        int ax = axis;
        if (ax < 0) ax += ndim;
        INS_CHECK(ndim >= 1, "lexsort: keys must be at least 1D");
        INS_CHECK(ax >= 0 && ax < ndim, "lexsort: axis out of range");

        // Move target axis to last position
        std::vector<int> perm(ndim);
        for (int i = 0; i < ndim; ++i) perm[i] = i;
        if (ax != ndim - 1) {
            std::swap(perm[ax], perm[ndim - 1]);
        }

        Array transposed = keys.transpose(perm);
        transposed = transposed.contiguous();

        const Shape& trans_shape = transposed.shape();

        // Determine batch dimensions (all except last axis)
        int64_t last_dim = trans_shape.dim(ndim - 1);
        int64_t batch_size = 1;
        for (int i = 0; i < ndim - 1; ++i) {
            batch_size *= trans_shape.dim(i);
        }

        // Number of keys = total_size / (batch_size * last_dim)
        int64_t total = transposed.numel();
        int64_t nkeys = total / (batch_size * last_dim);

        Array result(keys.shape(), DType::I64, keys.place());

        ops().launch("lexsort", keys.place(), keys.dtype(),
            { (void*)result.layout_ptr(), transposed.layout_ptr(),
              &batch_size, &last_dim, &nkeys },
            { (void*)result.layout_ptr() });

        // Transpose back if needed
        if (ax != ndim - 1) {
            std::vector<int> inv_perm(ndim);
            for (int i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;
            result = result.transpose(inv_perm);
        }

        return result;
    }

    // ============================================================================
    // indices
    // ============================================================================

    /**
     * @brief Return an array representing the indices of a grid.
     *
     * @param shape Shape of the grid
     * @param sparse If true, return sparse representation (not implemented)
     * @return Dense grid indices array of shape (ndim, dim0, dim1, ...)
     */
    Array indices(const Shape& shape, bool sparse) {
        int ndim = shape.ndim();
        if (sparse) {
            INS_THROW("indices: sparse mode not yet implemented - use dense mode");
        }

        // Dense mode: output shape is (ndim, dim0, dim1, ...)
        std::vector<int64_t> out_dims;
        out_dims.push_back(ndim);
        for (int i = 0; i < ndim; ++i) {
            out_dims.push_back(shape.dim(i));
        }
        Shape out_shape(out_dims);

        Array result(out_shape, DType::I64, CPUPlace());

        // Convert shape to int64_t array for kernel
        std::vector<int64_t> shape_dims = shape.dims();
        int64_t* shape_ptr = shape_dims.data();
        int ndim_val = ndim;

        ops().launch("indices", CPUPlace(), DType::I64,
            { (void*)result.layout_ptr(), (void*)&ndim_val, (void*)shape_ptr },
            { (void*)result.layout_ptr() });

        return result;
    }

    // ============================================================================
    // ix_
    // ============================================================================

    /**
     * @brief Construct an open mesh from multiple sequences.
     *
     * @param arrays 1D arrays to form the mesh
     * @return Vector of broadcasted arrays for each dimension
     */
    std::vector<Array> ix_(const std::vector<Array>& arrays) {
        int n = static_cast<int>(arrays.size());
        if (n == 0) return {};

        for (const auto& arr : arrays) {
            INS_CHECK(arr.shape().ndim() == 1, "ix_: all inputs must be 1D");
        }

        // Build broadcast shapes: each array gets shape with 1s except its own dimension
        std::vector<Shape> target_shapes(n);
        for (int i = 0; i < n; ++i) {
            std::vector<int64_t> dims(n, 1);
            dims[i] = arrays[i].numel();
            target_shapes[i] = Shape(dims);
        }

        std::vector<Array> result;
        for (int i = 0; i < n; ++i) {
            Array reshaped = arrays[i].reshape(target_shapes[i]);
            result.push_back(reshaped);
        }
        return result;
    }

    // ============================================================================
    // interp (linear interpolation)
    // ============================================================================

    /**
     * @brief Extract a scalar value from an array at a given index.
     *
     * @param arr Input array
     * @param idx Index
     * @return Double value
     */
    static double extract_scalar(const Array& arr, int64_t idx) {
        Array scalar = arr.at(idx);
        if (scalar.dtype() == DType::F32) {
            return static_cast<double>(scalar.item<float>());
        }
        else if (scalar.dtype() == DType::F64) {
            return scalar.item<double>();
        }
        else {
            INS_THROW("extract_scalar: unsupported dtype ", dtype_name(scalar.dtype()));
        }
    }

    /**
     * @brief One-dimensional linear interpolation.
     *
     * @param x The x-coordinates at which to evaluate the interpolated values
     * @param xp The x-coordinates of the data points
     * @param fp The y-coordinates of the data points
     * @param left Value to return for x < xp[0] (default: fp[0])
     * @param right Value to return for x > xp[-1] (default: fp[-1])
     * @return Interpolated values
     */
    Array interp(const Array& x, const Array& xp, const Array& fp,
        std::optional<double> left, std::optional<double> right) {
        INS_CHECK(x.defined() && xp.defined() && fp.defined(),
            "interp: inputs are undefined");
        INS_CHECK(xp.shape().ndim() == 1, "interp: xp must be 1D");
        INS_CHECK(fp.shape().ndim() == 1, "interp: fp must be 1D");
        INS_CHECK(xp.numel() == fp.numel(), "interp: xp and fp must have same length");

        // Ensure xp is sorted
        Array sort_idx = argsort(xp);
        Array sorted_xp = take(xp, sort_idx);
        Array sorted_fp = take(fp, sort_idx);

        double left_val = left.has_value() ? left.value() : extract_scalar(sorted_fp, 0);
        double right_val = right.has_value() ? right.value() : extract_scalar(sorted_fp, -1);
        double xp_min = extract_scalar(sorted_xp, 0);
        double xp_max = extract_scalar(sorted_xp, -1);

        // Find insertion indices
        Array right_idxs = searchsorted(sorted_xp, x, "right");
        Array left_idxs = sub(right_idxs, Array(1));

        // Clamp indices
        Array safe_left = maximum(left_idxs, zeros_like(left_idxs));
        Array safe_right = minimum(right_idxs, full(right_idxs.shape(), sorted_xp.numel() - 1, DType::I64));

        // Get values at indices
        Array x_left = take(sorted_xp, safe_left);
        Array x_right = take(sorted_xp, safe_right);
        Array y_left = take(sorted_fp, safe_left);
        Array y_right = take(sorted_fp, safe_right);

        // Linear interpolation with division by zero handling
        Array denominator = sub(x_right, x_left);
        Array numerator = sub(x, x_left);

        // Use isnan to handle division by zero
        // When denominator == 0, t will be NaN, then we replace with 0
        Array t = div(numerator, denominator);
        t = where(isnan(t), zeros_like(t), t);

        Array interp_vals = add(y_left, mul(t, sub(y_right, y_left)));

        // Handle boundaries
        Array result = where(less_equal(x, full(x.shape(), xp_min, interp_vals.dtype())),
            full(x.shape(), left_val, interp_vals.dtype()),
            interp_vals);
        result = where(greater_equal(x, full(x.shape(), xp_max, interp_vals.dtype())),
            full(x.shape(), right_val, interp_vals.dtype()),
            result);

        // Replace any remaining NaN values with left_val or right_val
        // (safety net for edge cases)
        result = where(isnan(result),
            where(less_equal(x, full(x.shape(), xp_min, result.dtype())),
                full(x.shape(), left_val, result.dtype()),
                full(x.shape(), right_val, result.dtype())),
            result);

        return result;
    }

} // namespace ins
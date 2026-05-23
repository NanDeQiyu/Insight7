// include/insight/c_api/array.h
#pragma once
#include "insight/c_api/dtype.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INSIGHT_MAX_NDIM 10

    /**
     * @brief Array metadata and lifecycle management.
     *
     * Data pointer and memory management are handled externally via Place.
     * This structure only manages metadata and reference counting for shared
     * data ownership across multiple Array instances.
     */
    typedef struct {
        void* data;                              // Data pointer (owned externally)
        int64_t dims[INSIGHT_MAX_NDIM];          // Dimension sizes
        int64_t strides[INSIGHT_MAX_NDIM];       // Strides for each dimension
        int32_t ndim;                            // Number of dimensions
        int64_t numel;                           // Total number of elements
        int32_t dtype;                           // InsightDType enum value
        int32_t device_type;                     // INSIGHT_DEVICE_CPU or INSIGHT_DEVICE_GPU
        int32_t device_id;                       // Physical device ID
        int64_t offset;                          // Element offset for views
        int32_t is_view;                         // 0 = owns data, 1 = view
        int32_t* ref_count;                      // Reference counter (shared across views)
    } InsightArray;

    /**
     * @brief Initialize an Array with newly allocated data.
     *
     * The data pointer is provided by the caller (allocated via Place::allocate).
     * This function initializes metadata and sets reference count to 1.
     *
     * @param array       Pointer to uninitialized InsightArray
     * @param data        Pre-allocated data pointer (can be NULL for empty arrays)
     * @param dims        Dimension array
     * @param ndim        Number of dimensions
     * @param dtype       Data type
     * @param device_type Device type (CPU or GPU)
     * @param device_id   Physical device ID
     * @return C_SUCCESS on success
     */
    C_Status insight_array_create(InsightArray* array,
        void* data,
        const int64_t* dims, int32_t ndim,
        int32_t dtype,
        int32_t device_type, int32_t device_id);

    /**
     * @brief Destroy an Array and release shared data if this is the last reference.
     *
     * Decrements the reference count. If it reaches zero, the reference counter
     * is freed. The caller is responsible for deallocating the data pointer
     * (via Place::deallocate) before calling this function.
     *
     * @param array Pointer to the Array to destroy
     * @return C_SUCCESS on success
     */
    C_Status insight_array_destroy(InsightArray* array);

    /**
     * @brief Create a view of an existing Array.
     *
     * Shares the same data pointer and reference count. No data copy is made.
     * The parent Array must outlive all views created from it.
     *
     * @param dst        Pointer to uninitialized InsightArray for the view
     * @param src        Parent Array to create a view from
     * @param offset     Element offset into the parent's data
     * @param dims       View dimension array
     * @param ndim       Number of view dimensions
     * @param strides    View strides array
     * @return C_SUCCESS on success
     */
    C_Status insight_array_create_view(InsightArray* dst,
        const InsightArray* src,
        int64_t offset,
        const int64_t* dims, int32_t ndim,
        const int64_t* strides);

    /**
     * @brief Get the total number of elements in the Array.
     */
    int64_t insight_array_numel(const InsightArray* array);

    /**
     * @brief Get the byte size of the Array's data.
     */
    size_t insight_array_nbytes(const InsightArray* array);

    /**
     * @brief Check if the Array is contiguous in memory.
     */
    int insight_array_is_contiguous(const InsightArray* array);

#ifdef __cplusplus
}
#endif
// include/insight/c_api/shape.h
#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define INSIGHT_MAX_NDIM 10

typedef struct {
  int64_t dims[INSIGHT_MAX_NDIM];
  int32_t ndim;
  int64_t numel;
} InsightShape;

// 查询
int64_t insight_shape_dim(const InsightShape *shape,
                          int32_t axis); // support negative indexing
int64_t insight_shape_numel(const InsightShape *shape);

#ifdef __cplusplus
}
#endif
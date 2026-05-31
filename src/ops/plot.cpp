// src/ops/plot.cpp
#ifdef INSIGHT_USE_MATPLOT

#include "insight/ops/plot.h"
#include "insight/core/exception.h"
#include <matplot/matplot.h>

namespace ins {
namespace plot {

namespace mp = matplot;

// ============================================================================
// Helpers
// ============================================================================

vector_1d to_vector(const Array &a) {
  INS_CHECK(a.defined(), "plot: array is undefined");
  Array cpu_a = a;
  if (a.place().kind() != DeviceKind::CPU)
    cpu_a = a.to(CPUPlace());
  int64_t n = cpu_a.numel();
  vector_1d result(n);
  switch (cpu_a.dtype()) {
  case DType::F64: {
    std::copy(cpu_a.data<double>(), cpu_a.data<double>() + n, result.begin());
    break;
  }
  case DType::F32: {
    const float *d = cpu_a.data<float>();
    for (int64_t i = 0; i < n; ++i)
      result[i] = static_cast<double>(d[i]);
    break;
  }
  case DType::I64: {
    const int64_t *d = cpu_a.data<int64_t>();
    for (int64_t i = 0; i < n; ++i)
      result[i] = static_cast<double>(d[i]);
    break;
  }
  case DType::I32: {
    const int32_t *d = cpu_a.data<int32_t>();
    for (int64_t i = 0; i < n; ++i)
      result[i] = static_cast<double>(d[i]);
    break;
  }
  default: {
    Array f64 = cpu_a.to(DType::F64);
    std::copy(f64.data<double>(), f64.data<double>() + n, result.begin());
  }
  }
  return result;
}

vector_2d to_matrix(const Array &a) {
  INS_CHECK(a.defined(), "plot: array is undefined");
  INS_CHECK(a.shape().ndim() == 2, "plot: array must be 2D");
  int64_t rows = a.shape().dim(0), cols = a.shape().dim(1);
  vector_1d flat = to_vector(a);
  vector_2d result(rows, vector_1d(cols));
  for (int64_t i = 0; i < rows; ++i)
    for (int64_t j = 0; j < cols; ++j)
      result[i][j] = flat[i * cols + j];
  return result;
}

// ============================================================================
// Figure & Axes
// ============================================================================

void figure(int n) { n < 0 ? mp::figure() : mp::figure(n); }
void subplot(int r, int c, int i) { mp::subplot(r, c, i); }
void hold(bool on) { mp::hold(on); }
void clf() { mp::cla(); }
void show() { mp::show(); }
void save(const std::string &f) { mp::gcf()->save(f); }

// ============================================================================
// Line Plots
// ============================================================================

void plot(const Array &y, const std::string &fmt) {
  auto v = to_vector(y);
  fmt.empty() ? (void)mp::plot(v) : (void)mp::plot(v, fmt);
}
void plot(const Array &x, const Array &y, const std::string &fmt) {
  auto xv = to_vector(x), yv = to_vector(y);
  fmt.empty() ? (void)mp::plot(xv, yv) : (void)mp::plot(xv, yv, fmt);
}
void plot(const Array &x1, const Array &y1, const std::string &f1,
          const Array &x2, const Array &y2, const std::string &f2) {
  mp::plot(to_vector(x1), to_vector(y1), f1, to_vector(x2), to_vector(y2), f2);
}
void plot3(const Array &x, const Array &y, const Array &z,
           const std::string &fmt) {
  auto ax = mp::gca();
  fmt.empty() ? (void)ax->plot3(to_vector(x), to_vector(y), to_vector(z))
              : (void)ax->plot3(to_vector(x), to_vector(y), to_vector(z), fmt);
}
void fplot(std::function<double(double)> f, double xmin, double xmax,
           const std::string &fmt, int n) {
  vector_1d x(n), y(n);
  double dx = (xmax - xmin) / (n - 1);
  for (int i = 0; i < n; ++i) {
    x[i] = xmin + i * dx;
    y[i] = f(x[i]);
  }
  fmt.empty() ? (void)mp::plot(x, y) : (void)mp::plot(x, y, fmt);
}
void fplot3(std::function<double(double)> fx, std::function<double(double)> fy,
            std::function<double(double)> fz, double tmin, double tmax,
            const std::string &fmt, int n) {
  vector_1d x(n), y(n), z(n);
  double dt = (tmax - tmin) / (n - 1);
  for (int i = 0; i < n; ++i) {
    double t = tmin + i * dt;
    x[i] = fx(t);
    y[i] = fy(t);
    z[i] = fz(t);
  }
  auto ax = mp::gca();
  fmt.empty() ? (void)ax->plot3(x, y, z) : (void)ax->plot3(x, y, z, fmt);
}
void fimplicit(const std::string &, double, double) {
  // matplotplusplus fimplicit requires function, not string
}
void fimplicit(std::function<double(double, double)> f, double, double) {
  mp::fimplicit(f);
}
void stairs(const Array &y, const std::string &fmt) {
  auto v = to_vector(y);
  fmt.empty() ? (void)mp::stairs(v) : (void)mp::stairs(v, fmt);
}
void stairs(const Array &x, const Array &y, const std::string &fmt) {
  auto xv = to_vector(x), yv = to_vector(y);
  fmt.empty() ? (void)mp::stairs(xv, yv) : (void)mp::stairs(xv, yv, fmt);
}
void errorbar(const Array &x, const Array &y, const Array &err,
              const std::string &fmt) {
  auto ax = mp::gca();
  fmt.empty()
      ? (void)ax->errorbar(to_vector(x), to_vector(y), to_vector(err))
      : (void)ax->errorbar(to_vector(x), to_vector(y), to_vector(err), fmt);
}
void area(const Array &y) { mp::area(to_vector(y)); }
void area(const Array &x, const Array &y) {
  mp::area(to_vector(x), to_vector(y));
}
void fill(const Array &x, const Array &y, const std::string &color) {
  mp::fill(to_vector(x), to_vector(y), color);
}
void line(const Array &, const Array &, const std::string &) {
  // matplotplusplus line() template forwarding issue
}

// ============================================================================
// Logarithmic Plots
// ============================================================================

void semilogx(const Array &y, const std::string &fmt) {
  auto v = to_vector(y);
  fmt.empty() ? (void)mp::semilogx(v) : (void)mp::semilogx(v, fmt);
}
void semilogx(const Array &x, const Array &y, const std::string &fmt) {
  auto xv = to_vector(x), yv = to_vector(y);
  fmt.empty() ? (void)mp::semilogx(xv, yv) : (void)mp::semilogx(xv, yv, fmt);
}
void semilogy(const Array &y, const std::string &fmt) {
  auto v = to_vector(y);
  fmt.empty() ? (void)mp::semilogy(v) : (void)mp::semilogy(v, fmt);
}
void semilogy(const Array &x, const Array &y, const std::string &fmt) {
  auto xv = to_vector(x), yv = to_vector(y);
  fmt.empty() ? (void)mp::semilogy(xv, yv) : (void)mp::semilogy(xv, yv, fmt);
}
void loglog(const Array &y, const std::string &fmt) {
  auto v = to_vector(y);
  fmt.empty() ? (void)mp::loglog(v) : (void)mp::loglog(v, fmt);
}
void loglog(const Array &x, const Array &y, const std::string &fmt) {
  auto xv = to_vector(x), yv = to_vector(y);
  fmt.empty() ? (void)mp::loglog(xv, yv) : (void)mp::loglog(xv, yv, fmt);
}

// ============================================================================
// Scatter
// ============================================================================

void scatter(const Array &x, const Array &y, double sz) {
  mp::scatter(to_vector(x), to_vector(y), sz);
}
void scatter3(const Array &x, const Array &y, const Array &z, double) {
  mp::scatter3(to_vector(x), to_vector(y), to_vector(z), "o");
}
void binscatter(const Array &, const Array &) {
  // matplotplusplus binscatter template forwarding issue
}

// ============================================================================
// Bar Charts
// ============================================================================

void bar(const Array &y, double w) { mp::bar(to_vector(y), w); }
void bar(const Array &x, const Array &y, double w) {
  (void)w;
  mp::bar(to_vector(x), to_vector(y));
}
void barh(const Array &y, double w) {
  (void)w;
  mp::bar(to_vector(y));
}
void barh(const Array &x, const Array &y, double w) {
  (void)w;
  mp::bar(to_vector(x), to_vector(y));
}
void barstacked(const Array &Y) { mp::barstacked(to_matrix(Y)); }
void barstacked(const Array &x, const Array &Y) {
  mp::barstacked(to_vector(x), to_matrix(Y));
}
void pareto(const Array &y) { mp::pareto(to_vector(y)); }
void pareto(const Array &, const Array &) {
  // matplotplusplus pareto(x,y) template forwarding issue
}

// ============================================================================
// Histograms & Stats
// ============================================================================

void hist(const Array &data, int bins) { mp::hist(to_vector(data), bins); }
void hist2(const Array &x, const Array &y, int) {
  mp::gca()->binscatter(to_vector(x), to_vector(y));
}
void boxplot(const Array &data) { mp::boxplot(to_vector(data)); }
void boxplot(const Array &data, const std::vector<std::string> &groups) {
  mp::boxplot(to_vector(data), groups);
}
void heatmap(const Array &data) { mp::heatmap(to_matrix(data)); }

// ============================================================================
// Stem
// ============================================================================

void stem(const Array &y, const std::string &fmt) {
  auto v = to_vector(y);
  fmt.empty() ? (void)mp::stem(v) : (void)mp::stem(v, fmt);
}
void stem(const Array &x, const Array &y, const std::string &fmt) {
  auto xv = to_vector(x), yv = to_vector(y);
  fmt.empty() ? (void)mp::stem(xv, yv) : (void)mp::stem(xv, yv, fmt);
}
void stem3(const Array &x, const Array &y, const Array &z,
           const std::string &fmt) {
  auto ax = mp::gca();
  fmt.empty() ? (void)ax->stem3(to_vector(x), to_vector(y), to_vector(z))
              : (void)ax->stem3(to_vector(x), to_vector(y), to_vector(z), fmt);
}

// ============================================================================
// Contour
// ============================================================================

void contour(const Array &X, const Array &Y, const Array &Z, int lv) {
  mp::contour(to_matrix(X), to_matrix(Y), to_matrix(Z), lv);
}
void contourf(const Array &X, const Array &Y, const Array &Z, int lv) {
  mp::contourf(to_matrix(X), to_matrix(Y), to_matrix(Z), lv);
}
void fcontour(const std::string &, double, double, double, double) {
  // matplotplusplus fcontour string overload not available
}
void fcontour(std::function<double(double, double)> f, double, double, double,
              double) {
  mp::fcontour(f);
}
void pcolor(const Array &C) { mp::pcolor(to_matrix(C)); }

// ============================================================================
// Surface / Mesh
// ============================================================================

void surf(const Array &X, const Array &Y, const Array &Z) {
  mp::surf(to_matrix(X), to_matrix(Y), to_matrix(Z));
}
void surfc(const Array &X, const Array &Y, const Array &Z) {
  mp::surfc(to_matrix(X), to_matrix(Y), to_matrix(Z));
}
void mesh(const Array &X, const Array &Y, const Array &Z) {
  mp::mesh(to_matrix(X), to_matrix(Y), to_matrix(Z));
}
void meshc(const Array &X, const Array &Y, const Array &Z) {
  mp::meshc(to_matrix(X), to_matrix(Y), to_matrix(Z));
}
void meshz(const Array &X, const Array &Y, const Array &Z) {
  mp::meshz(to_matrix(X), to_matrix(Y), to_matrix(Z));
}
void waterfall(const Array &X, const Array &Y, const Array &Z) {
  mp::waterfall(to_matrix(X), to_matrix(Y), to_matrix(Z));
}
void fsurf(std::function<double(double, double)> f,
           const std::vector<double> &) {
  mp::fsurf(f);
}
void fmesh(std::function<double(double, double)> f,
           const std::vector<double> &) {
  mp::fmesh(f);
}
void ribbon(const Array &) {
  // matplotplusplus ribbon template forwarding issue
}
void ribbon(const Array &, const Array &) {
  // matplotplusplus ribbon template forwarding issue
}
void fence(const Array &X, const Array &Y, const Array &Z) {
  mp::fence(to_matrix(X), to_matrix(Y), to_matrix(Z));
}

// ============================================================================
// Vector Fields
// ============================================================================

void quiver(const Array &X, const Array &Y, const Array &U, const Array &V,
            double) {
  mp::quiver(to_matrix(X), to_matrix(Y), to_matrix(U), to_matrix(V));
}
void quiver3(const Array &X, const Array &Y, const Array &Z, const Array &U,
             const Array &V, const Array &W) {
  mp::quiver3(to_matrix(X), to_matrix(Y), to_matrix(Z), to_matrix(U),
              to_matrix(V), to_matrix(W));
}
void feather(const Array &u, const Array &v) {
  mp::feather(to_vector(u), to_vector(v));
}

// ============================================================================
// Polar
// ============================================================================

void polarplot(const Array &theta, const Array &rho, const std::string &fmt) {
  auto tv = to_vector(theta), rv = to_vector(rho);
  fmt.empty() ? (void)mp::polarplot(tv, rv) : (void)mp::polarplot(tv, rv, fmt);
}
void polarscatter(const Array &theta, const Array &rho, double) {
  mp::polarscatter(to_vector(theta), to_vector(rho));
}
void polarhistogram(const Array &data, int bins) {
  mp::polarhistogram(to_vector(data), bins);
}
void compass(const Array &u, const Array &v) {
  mp::compass(to_vector(u), to_vector(v));
}
void ezpolar(const std::string &expr) { mp::ezpolar(expr); }

// ============================================================================
// Pie
// ============================================================================

void pie(const Array &x) { mp::pie(to_vector(x)); }
void pie(const Array &x, const std::vector<std::string> &labels) {
  mp::pie(to_vector(x), labels);
}

// ============================================================================
// Images
// ============================================================================

void imshow(const Array &data) { mp::imshow(to_matrix(data)); }
void image(const Array &data) { mp::image(to_matrix(data)); }
void imagesc(const Array &data) { mp::imagesc(to_matrix(data)); }

// ============================================================================
// Graph
// ============================================================================

void graph(const std::vector<int> &s, const std::vector<int> &t) {
  std::vector<size_t> ss(s.begin(), s.end()), tt(t.begin(), t.end());
  mp::graph(ss, tt);
}
void digraph(const std::vector<int> &s, const std::vector<int> &t) {
  std::vector<size_t> ss(s.begin(), s.end()), tt(t.begin(), t.end());
  mp::digraph(ss, tt);
}

// ============================================================================
// Parallel / PlotMatrix / Wordcloud / RGB
// ============================================================================

void parallelplot(const Array &data) { mp::parallelplot(to_matrix(data)); }
void plotmatrix(const Array &X) { mp::plotmatrix(to_matrix(X)); }
void wordcloud(const std::vector<std::string> &words,
               const std::vector<double> &sizes) {
  mp::wordcloud(words, sizes);
}
void rgbplot(const Array &data) { mp::rgbplot(to_matrix(data)); }

// ============================================================================
// Labels & Titles
// ============================================================================

void title(const std::string &t) { mp::title(t); }
void subtitle(const std::string &t) { mp::sgtitle(t); }
void xlabel(const std::string &t) { mp::xlabel(t); }
void ylabel(const std::string &t) { mp::ylabel(t); }
void zlabel(const std::string &t) { mp::zlabel(t); }
void legend(const std::vector<std::string> &l) { mp::legend(l); }

// ============================================================================
// Axis
// ============================================================================

void xlim(double a, double b) { mp::xlim({a, b}); }
void ylim(double a, double b) { mp::ylim({a, b}); }
void zlim(double a, double b) { mp::zlim({a, b}); }
void axis_equal() { mp::axis(mp::equal); }
void axis_tight() { mp::axis(mp::tight); }
void axis_off() { mp::axis(mp::off); }
void axis_ij() { mp::axis(mp::ij); }
void grid(bool on) { mp::grid(on); }

// ============================================================================
// Ticks
// ============================================================================

void xticks(const std::vector<double> &t) { mp::xticks(t); }
void yticks(const std::vector<double> &t) { mp::yticks(t); }
void zticks(const std::vector<double> &t) { mp::zticks(t); }
void xticklabels(const std::vector<std::string> &l) { mp::xticklabels(l); }
void yticklabels(const std::vector<std::string> &l) { mp::yticklabels(l); }
void xtickangle(double a) { mp::xtickangle(a); }
void ytickangle(double a) { mp::ytickangle(a); }

// ============================================================================
// Colormaps
// ============================================================================

void colormap(Colormap c) {
  switch (c) {
  case Colormap::Parula:
    mp::colormap(mp::palette::parula());
    break;
  case Colormap::Jet:
    mp::colormap(mp::palette::jet());
    break;
  case Colormap::HSV:
    mp::colormap(mp::palette::hsv());
    break;
  case Colormap::Hot:
    mp::colormap(mp::palette::hot());
    break;
  case Colormap::Cool:
    mp::colormap(mp::palette::cool());
    break;
  case Colormap::Spring:
    mp::colormap(mp::palette::spring());
    break;
  case Colormap::Summer:
    mp::colormap(mp::palette::summer());
    break;
  case Colormap::Autumn:
    mp::colormap(mp::palette::autumn());
    break;
  case Colormap::Winter:
    mp::colormap(mp::palette::winter());
    break;
  case Colormap::Bone:
    mp::colormap(mp::palette::bone());
    break;
  case Colormap::Copper:
    mp::colormap(mp::palette::copper());
    break;
  case Colormap::Pink:
    mp::colormap(mp::palette::pink());
    break;
  case Colormap::Lines:
    mp::colormap(mp::palette::lines());
    break;
  case Colormap::Colorcube:
    mp::colormap(mp::palette::colorcube());
    break;
  case Colormap::Prism:
    mp::colormap(mp::palette::prism());
    break;
  case Colormap::Flag:
    mp::colormap(mp::palette::flag());
    break;
  }
}
void colorbar(bool on) { mp::colorbar(on); }

// ============================================================================
// Annotations & Shapes
// ============================================================================

void text(double x, double y, const std::string &s) { mp::text(x, y, s); }
void text(const std::vector<double> &x, const std::vector<double> &y,
          const std::string &s) {
  mp::text(x, y, s);
}
void arrow(double x1, double y1, double x2, double y2) {
  mp::arrow(x1, y1, x2, y2);
}
void rectangle(double x, double y, double w, double h) {
  mp::rectangle(x, y, w, h);
}
void ellipse(double cx, double cy, double rx, double ry) {
  mp::ellipse(cx, cy, rx, ry);
}

// ============================================================================
// 3D Camera
// ============================================================================

void view(double az, double el) { mp::view(az, el); }

} // namespace plot
} // namespace ins

#endif // INSIGHT_USE_MATPLOT

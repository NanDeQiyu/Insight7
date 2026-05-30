// insight/ops/plot.h
// Complete matplotlib-equivalent plotting API wrapping matplotplusplus.
// All functions accept ins::Array and auto-convert to std::vector<double>.
#pragma once

#ifdef INSIGHT_USE_MATPLOT

#include "insight/core/array.h"
#include <functional>
#include <string>
#include <vector>

namespace ins {
namespace plot {

// ============================================================================
// Type aliases & Helpers
// ============================================================================
using vector_1d = std::vector<double>;
using vector_2d = std::vector<std::vector<double>>;

vector_1d to_vector(const Array &a);
vector_2d to_matrix(const Array &a);

// ============================================================================
// Figure & Axes Management
// ============================================================================
void figure(int number = -1);
void subplot(int rows, int cols, int index);
void hold(bool on);
void clf();
void show();
void save(const std::string &filename);

// ============================================================================
// Line Plots
// ============================================================================
void plot(const Array &y, const std::string &format = "");
void plot(const Array &x, const Array &y, const std::string &format = "");
void plot(const Array &x1, const Array &y1, const std::string &fmt1,
          const Array &x2, const Array &y2, const std::string &fmt2 = "");
void plot3(const Array &x, const Array &y, const Array &z,
           const std::string &format = "");
void fplot(std::function<double(double)> f, double xmin, double xmax,
           const std::string &format = "", int n_points = 100);
void fplot3(std::function<double(double)> fx, std::function<double(double)> fy,
            std::function<double(double)> fz, double tmin, double tmax,
            const std::string &format = "", int n_points = 100);
void fimplicit(const std::string &expr, double xmin = -5, double xmax = 5);
void fimplicit(std::function<double(double, double)> f, double xmin = -5,
               double xmax = 5);
void stairs(const Array &y, const std::string &format = "");
void stairs(const Array &x, const Array &y, const std::string &format = "");
void errorbar(const Array &x, const Array &y, const Array &err,
              const std::string &format = "");
void area(const Array &y);
void area(const Array &x, const Array &y);
void fill(const Array &x, const Array &y, const std::string &color = "b");
void line(const Array &x, const Array &y, const std::string &format = "");

// ============================================================================
// Logarithmic Plots
// ============================================================================
void semilogx(const Array &y, const std::string &format = "");
void semilogx(const Array &x, const Array &y, const std::string &format = "");
void semilogy(const Array &y, const std::string &format = "");
void semilogy(const Array &x, const Array &y, const std::string &format = "");
void loglog(const Array &y, const std::string &format = "");
void loglog(const Array &x, const Array &y, const std::string &format = "");

// ============================================================================
// Scatter Plots
// ============================================================================
void scatter(const Array &x, const Array &y, double size = 20.0);
void scatter3(const Array &x, const Array &y, const Array &z,
              double size = 20.0);
void binscatter(const Array &x, const Array &y);

// ============================================================================
// Bar Charts
// ============================================================================
void bar(const Array &y, double width = 0.8);
void bar(const Array &x, const Array &y, double width = 0.8);
void barh(const Array &y, double width = 0.8);
void barh(const Array &x, const Array &y, double width = 0.8);
void barstacked(const Array &Y);
void barstacked(const Array &x, const Array &Y);
void pareto(const Array &y);
void pareto(const Array &x, const Array &y);

// ============================================================================
// Histograms & Statistical Plots
// ============================================================================
void hist(const Array &data, int bins = 10);
void hist2(const Array &x, const Array &y, int bins = 10);
void boxplot(const Array &data);
void boxplot(const Array &data, const std::vector<std::string> &groups);
void heatmap(const Array &data);

// ============================================================================
// Stem Plots
// ============================================================================
void stem(const Array &y, const std::string &format = "");
void stem(const Array &x, const Array &y, const std::string &format = "");
void stem3(const Array &x, const Array &y, const Array &z,
           const std::string &format = "");

// ============================================================================
// Contour Plots
// ============================================================================
void contour(const Array &X, const Array &Y, const Array &Z, int levels = 10);
void contourf(const Array &X, const Array &Y, const Array &Z, int levels = 10);
void fcontour(const std::string &expr, double xmin = -5, double xmax = 5,
              double ymin = -5, double ymax = 5);
void fcontour(std::function<double(double, double)> f, double xmin = -5,
              double xmax = 5, double ymin = -5, double ymax = 5);
void pcolor(const Array &C);

// ============================================================================
// Surface / Mesh Plots
// ============================================================================
void surf(const Array &X, const Array &Y, const Array &Z);
void surfc(const Array &X, const Array &Y, const Array &Z);
void mesh(const Array &X, const Array &Y, const Array &Z);
void meshc(const Array &X, const Array &Y, const Array &Z);
void meshz(const Array &X, const Array &Y, const Array &Z);
void waterfall(const Array &X, const Array &Y, const Array &Z);
void fsurf(std::function<double(double, double)> f,
           const std::vector<double> &uv = {-5, 5, -5, 5});
void fmesh(std::function<double(double, double)> f,
           const std::vector<double> &uv = {-5, 5, -5, 5});
void ribbon(const Array &y);
void ribbon(const Array &x, const Array &y);
void fence(const Array &X, const Array &Y, const Array &Z);

// ============================================================================
// Vector Fields
// ============================================================================
void quiver(const Array &X, const Array &Y, const Array &U, const Array &V,
            double scale = 1.0);
void quiver3(const Array &X, const Array &Y, const Array &Z, const Array &U,
             const Array &V, const Array &W);
void feather(const Array &u, const Array &v);

// ============================================================================
// Polar Plots
// ============================================================================
void polarplot(const Array &theta, const Array &rho,
               const std::string &format = "");
void polarscatter(const Array &theta, const Array &rho, double size = 20.0);
void polarhistogram(const Array &data, int bins = 10);
void compass(const Array &u, const Array &v);
void ezpolar(const std::string &expr);

// ============================================================================
// Pie Charts
// ============================================================================
void pie(const Array &x);
void pie(const Array &x, const std::vector<std::string> &labels);

// ============================================================================
// Images
// ============================================================================
void imshow(const Array &data);
void image(const Array &data);
void imagesc(const Array &data);

// ============================================================================
// Graph / Network Plots
// ============================================================================
void graph(const std::vector<int> &sources, const std::vector<int> &targets);
void digraph(const std::vector<int> &sources, const std::vector<int> &targets);

// ============================================================================
// Parallel Coordinates
// ============================================================================
void parallelplot(const Array &data);

// ============================================================================
// Plot Matrix
// ============================================================================
void plotmatrix(const Array &X);

// ============================================================================
// Word Cloud
// ============================================================================
void wordcloud(const std::vector<std::string> &words,
               const std::vector<double> &sizes);

// ============================================================================
// RGB Plot
// ============================================================================
void rgbplot(const Array &data);

// ============================================================================
// Labels & Titles
// ============================================================================
void title(const std::string &text);
void subtitle(const std::string &text);
void xlabel(const std::string &text);
void ylabel(const std::string &text);
void zlabel(const std::string &text);
void legend(const std::vector<std::string> &labels);

// ============================================================================
// Axis Limits & Configuration
// ============================================================================
void xlim(double xmin, double xmax);
void ylim(double ymin, double ymax);
void zlim(double zmin, double zmax);
void axis_equal();
void axis_tight();
void axis_off();
void axis_ij();
void grid(bool on);

// ============================================================================
// Ticks
// ============================================================================
void xticks(const std::vector<double> &ticks);
void yticks(const std::vector<double> &ticks);
void zticks(const std::vector<double> &ticks);
void xticklabels(const std::vector<std::string> &labels);
void yticklabels(const std::vector<std::string> &labels);
void xtickangle(double angle);
void ytickangle(double angle);

// ============================================================================
// Colormaps
// ============================================================================
enum class Colormap {
  Parula,
  Jet,
  HSV,
  Hot,
  Cool,
  Spring,
  Summer,
  Autumn,
  Winter,
  Bone,
  Copper,
  Pink,
  Lines,
  Colorcube,
  Prism,
  Flag
};
void colormap(Colormap cmap);
void colorbar(bool on = true);

// ============================================================================
// Annotations & Shapes
// ============================================================================
void text(double x, double y, const std::string &str);
void text(const std::vector<double> &x, const std::vector<double> &y,
          const std::string &str);
void arrow(double x1, double y1, double x2, double y2);
void rectangle(double x, double y, double w, double h);
void ellipse(double cx, double cy, double rx, double ry);

// ============================================================================
// 3D Camera
// ============================================================================
void view(double azimuth, double elevation);

} // namespace plot
} // namespace ins

#endif // INSIGHT_USE_MATPLOT

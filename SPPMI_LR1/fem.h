#ifndef FEM_H
#define FEM_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <set>
#include <utility>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "solvers/CGM_LLT.h"

using namespace std;

// Класс точка
class point
{
public:
    union {double x; double r;};
    union {double y; double z;};
    size_t num;
    point();
    point(double x, double y);
    point(double x, double y, size_t num);
    point(const point & p);
    double & operator [] (size_t i);
    double operator [] (size_t i) const;
    bool operator < (const point & t) const;
    bool operator == (const point & t) const;
    point & operator = (const point & other);
    friend ostream & operator << (ostream & os, point a);
};

// Класс физическая область
class phys_area
{
public:
    double sigma;
    size_t num;
};

// Класс прямоугольник
class rectangle
{
public:
    point * nodes[4];
    phys_area * ph;
    void init();
    double get_local_G(size_t i, size_t j) const;
    double get_local_M(size_t i, size_t j) const;
    bool inside(const point & p) const;
    double bfunc_2d(size_t num, const point & p) const;
    point grad_bfunc2d(size_t num, const point & p) const;
private:
    point to_local(const point & p) const;
    point to_global(const point & p) const;
    point gauss_points[9];
    double gauss_weights[9];
    double jacobian;
    double hx, hy;
    double bfunc_1d_x(size_t num, double x) const;
    double bfunc_1d_y(size_t num, double y) const;
    double d_bfunc_1d_x(size_t num, double x) const;
    double d_bfunc_1d_y(size_t num, double y) const;
};

// Класс СЛАУ
class SLAE
{
public:
    SLAE();
    ~SLAE();
    void solve(double eps);
    void alloc_all(size_t n_size, size_t gg_size);
    void add(size_t i, size_t j, double elem);
    double * gg, * di, * rp, * x;
    size_t * ig, * jg;
    size_t n;
private:
    CGM_LLT solver;
};

// Класс МКЭ
class FEM
{
public:
    template<typename param_type>
    friend class inverse;

    FEM();
    ~FEM();
    SLAE slae;

    void input();
    void make_portrait();
    void assembling_global();
    void applying_bound();
    void applying_sources();

    double get_solution(const point & p) const;
    double get_solution(const point & p, const rectangle * fe) const;
    point get_grad(const point & p) const;
    point get_grad(const point & p, const rectangle * fe) const;
    void out_tecplot(const string & tecplot_filename) const;

private:
    point * nodes;
    size_t nodes_num;
    rectangle * rects;
    size_t rects_num;
    phys_area * phs;
    size_t phs_num;
    size_t pss_num;
    pair<point, double> * pss;

    const rectangle * get_fe(const point & p) const;
};

#endif // FEM_H

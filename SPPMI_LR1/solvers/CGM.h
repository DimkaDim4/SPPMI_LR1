#ifndef CGM_H_INCLUDED
#define CGM_H_INCLUDED

#include <cstdlib>

class CGM
{
public:
    void init(size_t * gi_s, size_t * gj_s, double * di_s,
              double * gg_s, size_t n_s);
    void solve(double * solution, double * rp, double gamma);

    CGM();
    ~CGM();
private:
    void mul_matrix(const double * f, double * x) const;
    double dot_prod(const double * a, const double * b) const;

    size_t n;
    size_t * gi, * gj;
    double * di, * gg;

    double * r, * w, * u, * s, * x0, * p, * t;
};

#endif // CGM_H_INCLUDED

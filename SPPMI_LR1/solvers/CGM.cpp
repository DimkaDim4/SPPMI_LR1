#include "CGM.h"
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <cstdio>
#include <cmath>

CGM::CGM()
{
    r = w = s = p = t = u = NULL;
}

CGM::~CGM()
{
    if(r) delete [] r;
    if(w) delete [] w;
    if(s) delete [] s;
    if(p) delete [] p;
    if(t) delete [] t;
    if(u) delete [] u;
}

void CGM::init(size_t * gi_s, size_t * gj_s, double * di_s,
                         double * gg_s, size_t n_s)
{
    gi = gi_s;
    gj = gj_s;
    di = di_s;
    gg = gg_s;
    n = n_s;

    r = new double [n];
    w = new double [n];
    s = new double [n];
    p = new double [n];
    t = new double [n];
    u = new double [n];
}

double CGM::dot_prod(const double * a, const double * b) const
{
    double res = 0.0;
    for(size_t i = 0; i < n; i++)
    {
        res += a[i] * b[i];
    }
    return res;
}

void CGM::mul_matrix(const double * f, double * x) const
{
    for(size_t i = 0; i < n; i++)
    {
        double v_el = f[i];
        x[i] = di[i] * v_el;
        for(size_t k = gi[i], k1 = gi[i + 1]; k < k1; k++)
        {
            size_t j = gj[k];
            x[i] += gg[k] * f[j];
            x[j] += gg[k] * v_el;
        }
    }
}

void CGM::solve(double * solution, double * rp, double gamma)
{
    double eps = gamma;
    size_t max_iter = /*(size_t)sqrt(n)*/ 15000;

    double alpha, beta, alpha1, alpha2;
    double rp_norm = sqrt(dot_prod(rp, rp));
    x0 = new double [n];

    mul_matrix(solution, t);
    for(size_t i = 0; i < n; i++)
    {
        x0[i] = solution[i];
        r[i] = rp[i] - t[i];
        w[i] = r[i] / di[i];
        p[i] = 0.0;
    }

    alpha1 = dot_prod(r, w);
    beta = 0.0;

    bool not_end = true;
    double discr;

    size_t iter;
    for(iter = 0; iter < max_iter && not_end; iter++)
    {
        discr = sqrt(dot_prod(r, r));

        //if(iter%10 == 0)
        {
            printf("CGM Residual:\t%5lu\t%.3e\r", (unsigned long)iter, discr / rp_norm);
            fflush(stdout);
        }

        if(discr / rp_norm > eps)
        {
            for(size_t i = 0; i < n; i++)
                p[i] = w[i] + beta * p[i];
            mul_matrix(p, u);
            alpha2 = dot_prod(u, p);
            alpha = alpha1 / alpha2;

            for(size_t i = 0; i < n; i++)
            {
                x0[i] += alpha * p[i];
                r[i] -= alpha * u[i];
                w[i] = r[i] / di[i];
            }

            alpha2 = alpha1;
            alpha1 = dot_prod(r, w);
            beta = alpha1 / alpha2;
        }
        else
            not_end = false;
    }

    mul_matrix(x0, r);
    for(size_t i = 0; i < n; i++)
        r[i] = rp[i] - r[i];
    discr = sqrt(dot_prod(r, r));
    printf("CGM Residual:\t%5lu\t%.3e\n", (unsigned long)iter, discr / rp_norm);

    if(iter >= max_iter)
        printf("Soulution can`t found, iteration limit exceeded!\n");

    for(size_t i = 0; i < n; i++)
        solution[i] = x0[i];
    delete [] x0;
}

#include "fem.h"

// ============================================================================

point::point()
{
    x = y = 0.0;
    num = 0;
}

point::point(double x, double y)
{
    this->x = x;
    this->y = y;
    num = 0;
}

point::point(double x, double y, size_t num)
{
    this->x = x;
    this->y = y;
    this->num = num;
}

point::point(const point & p)
{
    x = p.x;
    y = p.y;
    num = p.num;
}

double & point::operator [] (size_t i)
{
    switch(i)
    {
    case 0:
        return x;
    case 1:
        return y;
    default:
        stringstream str;
        str << "Error: Unknown index " << i << " at point " << * this << endl;
        throw str.str();
    }
}

double point::operator [] (size_t i) const
{
    switch(i)
    {
    case 0:
        return x;
    case 1:
        return y;
    default:
        stringstream str;
        str << "Error: Unknown index " << i << " at point " << * this << endl;
        throw str.str();
    }
}

bool point::operator < (const point & t) const
{
    return num < t.num;
}

bool point::operator == (const point & t) const
{
    return num == t.num;
}

point & point::operator = (const point & other)
{
    if(this != & other)
    {
        this->x = other.x;
        this->y = other.y;
        this->num = other.num;
    }
    return * this;
}

ostream & operator << (ostream & os, point a)
{
    os << "{ x = " << a.x << ", y = " << a.y << " }";
    return os;
}

// ============================================================================

point rectangle::to_local(const point & p) const
{
    double ksi = 2.0 * (p.x - nodes[0]->x) / hx - 1.0;
    double eta = 2.0 * (p.y - nodes[0]->y) / hy - 1.0;
    return point(ksi, eta);
}

point rectangle::to_global(const point & p) const
{
    double x = (p.x + 1.0) * hx / 2.0 + nodes[0]->x;
    double y = (p.y + 1.0) * hy / 2.0 + nodes[0]->y;
    return point(x, y);
}

void rectangle::init()
{
    // Расчет шага
    hx = nodes[1]->x - nodes[0]->x;
    hy = nodes[2]->y - nodes[0]->y;

    // Точки Гаусса в локальной с.к.
    double gauss_points_local[2][9] =
    {
        {0.0, 0.0, 0.0, sqrt(3.0 / 5.0), -sqrt(3.0 / 5.0), sqrt(3.0 / 5.0), sqrt(3.0 / 5.0), -sqrt(3.0 / 5.0), -sqrt(3.0 / 5.0)},
        {0.0, sqrt(3.0 / 5.0), -sqrt(3.0 / 5.0), 0.0, 0.0, sqrt(3.0 / 5.0), -sqrt(3.0 / 5.0), sqrt(3.0 / 5.0), -sqrt(3.0 / 5.0)}
    };

    // Веса Гаусса
    gauss_weights[0] = 64.0 / 81.0;
    gauss_weights[1] = gauss_weights[2] = gauss_weights[3] = gauss_weights[4] = 40.0 / 81.0;
    gauss_weights[5] = gauss_weights[6] = gauss_weights[7] = gauss_weights[8] = 25.0 / 81.0;

    // Перевод с мастер-элемента на текущий
    for(int i = 0; i < 9; i++)
        gauss_points[i] = to_global(point(gauss_points_local[0][i], gauss_points_local[1][i]));

    // Якобиан
    jacobian = hx * hy / 4.0;
}

double rectangle::bfunc_1d_x(size_t num, double x) const
{
    if(num == 0)
        return (nodes[3]->x - x) / hx;
    else if(num == 1)
        return (x - nodes[0]->x) / hx;

    stringstream str;
    str << "Error: Unknown number " << num << " at bfunc_1d_x" << endl;
    throw str.str();
    return 0.0;
}

double rectangle::bfunc_1d_y(size_t num, double y) const
{
    if(num == 0)
        return (nodes[3]->y - y) / hy;
    else if(num == 1)
        return (y - nodes[0]->y) / hy;

    stringstream str;
    str << "Error: Unknown number " << num << " at bfunc_1d_y" << endl;
    throw str.str();
    return 0.0;
}

double rectangle::d_bfunc_1d_x(size_t num, double x) const
{
    if(num == 0)
        return - 1.0 / hx;
    else if(num == 1)
        return 1.0 / hx;

    (void)x;
    stringstream str;
    str << "Error: Unknown number " << num << " at bfunc_1d_y" << endl;
    throw str.str();
    return 0.0;
}

double rectangle::d_bfunc_1d_y(size_t num, double y) const
{
    if(num == 0)
        return - 1.0 / hy;
    else if(num == 1)
        return 1.0 / hy;

    (void)y;
    stringstream str;
    str << "Error: Unknown number " << num << " at bfunc_1d_y" << endl;
    throw str.str();
    return 0.0;
}

double rectangle::bfunc_2d(size_t num, const point & p) const
{
    switch(num)
    {
    case 0:
        return bfunc_1d_x(0, p.x) * bfunc_1d_y(0, p.y);
    case 1:
        return bfunc_1d_x(1, p.x) * bfunc_1d_y(0, p.y);
    case 2:
        return bfunc_1d_x(0, p.x) * bfunc_1d_y(1, p.y);
    case 3:
        return bfunc_1d_x(1, p.x) * bfunc_1d_y(1, p.y);
    }

    stringstream str;
    str << "Error: Unknown number " << num << " at bfunc_2d" << endl;
    throw str.str();
    return 0.0;
}

point rectangle::grad_bfunc2d(size_t num, const point & p) const
{
    switch(num)
    {
    case 0:
        return point(d_bfunc_1d_x(0, p.x) * bfunc_1d_y(0, p.y),
                     bfunc_1d_x(0, p.x) * d_bfunc_1d_y(0, p.y));
    case 1:
        return point(d_bfunc_1d_x(1, p.x) * bfunc_1d_y(0, p.y),
                     bfunc_1d_x(1, p.x) * d_bfunc_1d_y(0, p.y));
    case 2:
        return point(d_bfunc_1d_x(0, p.x) * bfunc_1d_y(1, p.y),
                     bfunc_1d_x(0, p.x) * d_bfunc_1d_y(1, p.y));
    case 3:
        return point(d_bfunc_1d_x(1, p.x) * bfunc_1d_y(1, p.y),
                     bfunc_1d_x(1, p.x) * d_bfunc_1d_y(1, p.y));
    }

    stringstream str;
    str << "Error: Unknown number " << num << " at grad_bfunc2d" << endl;
    throw str.str();
    return point(0.0, 0.0);
}

double rectangle::get_local_G(size_t i, size_t j) const
{
    double result = 0.0;
    for(int g = 0; g < 9; g++)
    {
        point grad_i = grad_bfunc2d(i, gauss_points[g]);
        point grad_j = grad_bfunc2d(j, gauss_points[g]);
        result += gauss_weights[g] * gauss_points[g].r * // x это типа r
                (grad_i.r * grad_j.r + grad_i.z * grad_j.z) * ph->sigma;
    }
    result *= jacobian;
    return result;
}

double rectangle::get_local_M(size_t i, size_t j) const
{
    double result = 0.0;
    for(int g = 0; g < 9; g++)
    {
        result += gauss_weights[g] * gauss_points[g].r * // x это типа r
                bfunc_2d(i, gauss_points[g]) * bfunc_2d(j, gauss_points[g]);
    }
    result *= jacobian;
    return result;
}

bool rectangle::inside(const point & p) const
{
    if(nodes[0]->x <= p.x && nodes[0]->y <= p.y &&
       nodes[3]->x >= p.x && nodes[3]->y >= p.y)
        return true;
    return false;
}

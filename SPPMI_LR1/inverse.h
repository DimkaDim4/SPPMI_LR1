#ifndef INVERSE_H
#define INVERSE_H

#include "fem.h"

class sensor
{
public:
    point position;
    point value;
    const rectangle * rect;
    double trust;
};

class parameter_sigma
{
public:
    inline double get_value();
    inline void set_value(double val);
    inline static double get_mid();
    phys_area * phys;
};

class parameter_power
{
public:
    inline double get_value();
    inline void set_value(double val);
    inline static double get_mid();
    double power;
    pair<point, double> * pss;
};

template<typename param_type>
class inverse
{
public:
    FEM fem;

    inverse();
    ~inverse();
    void calc();

private:
    size_t sens_num;
    sensor * sens;

    size_t param_num;
    param_type * param;

    void precond();
    void read_sensors();
    void read_parameters();

    double ** A;
    double * b;
    double * c;
    void solve_gauss();

    void solve_fem();
    double get_residual();
    bool is_fpu_error(double x) const;
};

// ============================================================================

const double SLAE_EPSILON = 1e-16;
const double INV_EPSILON  = 1e-10;
const double DELTA = 0.005;
const double GAMMA = 1e-3;
const double POWER_MID = 1.0;
const double SIGMA_MID = 1.0;

// ============================================================================

inline double parameter_sigma::get_value()
{
    return phys->sigma;
}

inline void parameter_sigma::set_value(double val)
{
    phys->sigma = val;
}

inline double parameter_sigma::get_mid()
{
    return SIGMA_MID;
}

inline double parameter_power::get_value()
{
    return power;
}

inline void parameter_power::set_value(double val)
{
    power = val;
    // Пересчет мощности (КЭД)
    //pss[0].second = power / (2.0 * M_PI * pss[0].first.r);
    //pss[1].second = - power / (2.0 * M_PI * pss[1].first.r);
    // Пересчет мощности (ВЭЛ)
    pss[0].second = power;
    pss[1].second = -power;
}

inline double parameter_power::get_mid()
{
    return POWER_MID;
}

// Получение данных с датчиков
template<typename param_type>
void inverse<param_type>::read_sensors()
{
    // 2 приемника
/*
    sens_num = 2;
    sens = new sensor [sens_num];
    sens[0].position = point(50.0, 1e-5);
    sens[1].position = point(70.0, 1e-5);
    sens[0].trust = 1.0;
    sens[1].trust = 1.0;
    for(size_t i = 0; i < sens_num; i++)
    {
        sens[i].rect = fem.get_fe(sens[i].position);
        sens[i].value = fem.get_grad(sens[i].position, sens[i].rect);
    }
*/
    // 3 приемника

    sens_num = 3;
    sens = new sensor [sens_num];
    sens[0].position = point(50.0, 1e-5);
    sens[1].position = point(70.0, 1e-5);
    sens[2].position = point(90.0, 1e-5);
    sens[0].trust = 1.0;
    sens[1].trust = 1.0;
    sens[2].trust = 1.0;
    for(size_t i = 0; i < sens_num; i++)
    {
        sens[i].rect = fem.get_fe(sens[i].position);
        sens[i].value = fem.get_grad(sens[i].position, sens[i].rect);
    }

    // шум в первом приемнике
    double noise = 0.1;
    sens[0].value.r *= (1.0 + noise);
    sens[0].value.z *= (1.0 + noise);
    sens[0].trust = 0.1;

    //// шум во втором приемнике
    sens[1].value.r *= (1.0 + noise);
    sens[1].value.z *= (1.0 + noise);
    sens[1].trust = 0.1;

    //// шум в третьем приемнике
    sens[2].value.r *= (1.0 + noise);
    sens[2].value.z *= (1.0 + noise);
    sens[2].trust = 0.1;
}

// Получение неизвестных параметров сигма
template<>
void inverse<parameter_sigma>::read_parameters()
{
    param_num = 1;
    param = new parameter_sigma [param_num];
    param[0].phys = fem.phs;
    param[0].set_value(SIGMA_MID);
}

// Получение неизвестных параметров мощности
template<>
void inverse<parameter_power>::read_parameters()
{
    param_num = 1;
    param = new parameter_power [param_num];
    param[0].pss = fem.pss;
    param[0].set_value(POWER_MID);
}

// ============================================================================

template<typename param_type>
inverse<param_type>::inverse()
{
    sens_num = 0;
    sens = NULL;

    param_num = 0;
    param = NULL;

    A = NULL;
    b = NULL;
    c = NULL;
}

template<typename param_type>
inverse<param_type>::~inverse()
{
    if(sens) delete [] sens;
    if(param) delete [] param;
    if(b) delete [] b;
    if(c) delete [] c;
    if(A)
    {
        for(size_t i = 0; i < param_num; i++)
            delete [] A[i];
        delete [] A;
    }
}

// Решение обратной задачи
template<typename param_type>
void inverse<param_type>::calc()
{
    precond();
    read_sensors();
    read_parameters();
    solve_fem();
    cout.precision(17);

    // Матрица для обратной задачи
    A = new double * [param_num];
    for(size_t i = 0; i < param_num; i++)
        A[i] = new double [param_num];
    b = new double [param_num];
    c = new double [param_num];

    // Искомый параметр на предыдущей итерации
    double * old_params = new double [param_num];
    for(size_t i = 0; i < param_num; i++)
        old_params[i] = param[i].get_value();

    // Значения при приращении пар-ра (для вычисления производных)
    bool flag = true;
    point ** deriv = new point * [param_num];
    for(size_t i = 0; i < param_num; i++)
        deriv[i] = new point [sens_num];

    // Значения на предыдущей итерации
    vector<point> old_values;
    old_values.resize(sens_num);
    for(size_t i = 0; i < sens_num; i++)
        old_values[i] = fem.get_grad(sens[i].position, sens[i].rect);

    // Типа невязка?
    double residual = get_residual();

    // Цикл!
    for(size_t iter = 1; flag; iter++)
    {
        // Расчет новых значений при приращениях пар-ра
        for(size_t i = 0; i < param_num; i++)
        {
            double bak = param[i].get_value();
            param[i].set_value(bak * (1.0 + DELTA));
            solve_fem();
            for(size_t j = 0; j < sens_num; j++)
                deriv[i][j] = fem.get_grad(sens[j].position, sens[j].rect);
            param[i].set_value(bak);
        }

        // Заполнение основной матрицы СЛАУ и правой части (без альфы)
        memset(b, 0, sizeof(double) * param_num);
        for(size_t i = 0; i < param_num; i++)
        {
            memset(A[i], 0, sizeof(double) * param_num);
            for(size_t j = 0; j < param_num; j++)
            {
                for(size_t k = 0; k < sens_num; k++)
                {
                    double w2;
                    // Компонента r
                    w2 = (sens[k].trust * sens[k].trust) / (sens[k].value.r * sens[k].value.r);
                    A[i][j] += w2 *
                            (old_values[k].r - deriv[i][k].r) / (param[i].get_value() * DELTA) *
                            (old_values[k].r - deriv[j][k].r) / (param[j].get_value() * DELTA);
                    // Компонента z
                    w2 = (sens[k].trust * sens[k].trust) / (sens[k].value.z * sens[k].value.z);
                    A[i][j] += w2 *
                            (old_values[k].z - deriv[i][k].z) / (param[i].get_value() * DELTA) *
                            (old_values[k].z - deriv[j][k].z) / (param[j].get_value() * DELTA);
                }
            }

            for(size_t k = 0; k < sens_num; k++)
            {
                double w2;
                // Компонента r
                w2 = (sens[k].trust * sens[k].trust) / (sens[k].value.r * sens[k].value.r);
                b[i] -= w2 *
                        (sens[k].value.r - old_values[k].r) *
                        (old_values[k].r - deriv[i][k].r) / (param[i].get_value() * DELTA);
                // Компонента z
                w2 = (sens[k].trust * sens[k].trust) / (sens[k].value.z * sens[k].value.z);
                b[i] -= w2 *
                        (sens[k].value.z - old_values[k].z) *
                        (old_values[k].z - deriv[i][k].z) / (param[i].get_value() * DELTA);
            }
        }

        // Magic-magic
        double alpha = 0.0;
        for(size_t i = 0; i < param_num; i++)
        {
            double tmp = old_params[i] - param_type::get_mid();
            alpha += tmp * tmp;
        }
        if(alpha) alpha = GAMMA * residual / alpha;

        // Учет альфы
        for(size_t i = 0; i < param_num; i++)
        {
            A[i][i] += alpha;
            b[i] -= alpha * (old_params[i] - param_type::get_mid());
        }

        // Решаем СЛАУ
        solve_gauss();

        // Бетта
        double beta = 1.0;

        // Применение нового значения с учетом невязки
        double resid_new;
        while(beta > INV_EPSILON)
        {
            for(size_t i = 0; i < param_num; i++)
                param[i].set_value(old_params[i] + beta * c[i]);

            resid_new = get_residual();
            if(resid_new > residual || is_fpu_error(resid_new))
                beta /= 5.0;
            else
                break;
        }
        residual = resid_new;

        // Если не удалось подобрать бету - ну что, печалька
        if(beta <= INV_EPSILON)
        {
            cerr << "Stagnation detected, breaking." << endl;
            cerr << "Iterations: " << iter << endl;
            for(size_t i = 0; i < param_num; i++)
                cout << "param[" << i << "] = " << old_params[i] << endl;
            flag = false;
            break;
        }

        // Если решение не поменялось - тоже грустно
        size_t tmp_c = 0;
        for(size_t i = 0; i < param_num; i++)
            if(fabs((old_params[i] - param[i].get_value()) / old_params[i]) > INV_EPSILON)
                tmp_c++;
        if(tmp_c == 0)
        {
            cerr << "Stagnation detected, breaking." << endl;
            cerr << "Iterations: " << iter << endl;
            for(size_t i = 0; i < param_num; i++)
                cout << "param[" << i << "] = " << old_params[i] << endl;
            flag = false;
            break;
        }

        // Сохраняем старые значения
        solve_fem();
        for(size_t i = 0; i < sens_num; i++)
            old_values[i] = fem.get_grad(sens[i].position, sens[i].rect);
        for(size_t i = 0; i < param_num; i++)
            old_params[i] = param[i].get_value();

        // Нашли, ура
        if(residual <= INV_EPSILON)
        {
            cout << "Solution found, congratulations!" << endl;
            cout << "Iterations: " << iter << endl;
            for(size_t i = 0; i < param_num; i++)
                cout << "param[" << i << "] = " << old_params[i] << endl;
            flag = false;
        }

        for(size_t i = 0; i < param_num; i++)
            cout << "<" << iter << "> param[" << i << "] = " << old_params[i] << endl;
    }

    // Чистим память
    delete [] old_params;
    for(size_t i = 0; i < param_num; i++)
        delete [] deriv[i];
    delete [] deriv;
}

// Инициализация
template<typename param_type>
void inverse<param_type>::precond()
{
    fem.input();
    fem.make_portrait();
    solve_fem();
}

// Решение прямой задачи
template<typename param_type>
void inverse<param_type>::solve_fem()
{
    fem.assembling_global();
    fem.applying_sources();
    fem.applying_bound();
    fem.slae.solve(SLAE_EPSILON);
}

// Некошерные числа
template<typename param_type>
bool inverse<param_type>::is_fpu_error(double x) const
{
    double y = x - x;
    return x != x || y != y;
}

// Подсчет невязки
template<typename param_type>
double inverse<param_type>::get_residual()
{
    solve_fem();
    double residual = 0.0;
    for(size_t k = 0; k < sens_num; k++)
    {
        double w, tmp;
        point sol = fem.get_grad(sens[k].position, sens[k].rect);
        // Компонента r
        w = sens[k].trust / sens[k].value.r;
        tmp = w * (sens[k].value.r - sol.r);
        residual += tmp * tmp;
        // Компонента z
        w =  sens[k].trust / sens[k].value.z;
        tmp = w * (sens[k].value.z - sol.z);
        residual += tmp * tmp;
    }
    return residual;
}

// Метод Гаусса
template<typename param_type>
void inverse<param_type>::solve_gauss()
{
    int n = (int)param_num;
    //верхний треугольный вид
    for(int i = 0; i < n; i++)
    {
        if(!A[i][i])
        {
            bool flag = false;
            for(int j = i + 1; j < n && !flag; j++)
                if(A[j][i])
                {
                    for(int k = i; k < n; k++)
                    {
                        double tmp = A[i][k];
                        A[i][k] = A[j][k];
                        A[j][k] = tmp;
                    }
                    double tmp = b[i];
                    b[i] = b[j];
                    b[j] = tmp;
                    flag = true;
                }
        }
        b[i] = b[i] / A[i][i];
        for(int j = n - 1; j >= i; j--)
            A[i][j] = A[i][j] / A[i][i];
        for(int j = i + 1; j < n; j++)
        {
            b[j] -= b[i] * A[j][i];
            for(int k = n - 1; k >= i; k--)
                A[j][k] -= A[i][k] * A[j][i];
        }
    }
    //диагональный вид
    for(int i = n - 1; i > 0; i--)
        for(int j = i - 1; j >= 0; j--)
            b[j] -= A[j][i] * b[i];

    for(int i = 0; i < n; i++)
        c[i] = b[i];
}

#endif // INVERSE_H

#include "fem.h"

// ============================================================================

SLAE::SLAE()
{
    gg = di = rp = x = NULL;
    ig = jg = NULL;
    n = 0;
}

SLAE::~SLAE()
{
    if(gg) delete [] gg;
    if(di) delete [] di;
    if(rp) delete [] rp;
    if(x)  delete [] x;
    if(ig) delete [] ig;
    if(jg) delete [] jg;
}

void SLAE::solve(double eps)
{
    //cout << "Solving SLAE ..." << endl;
    solver.init(ig, jg, di, gg, n);
    solver.solve(x, rp, eps);
}

void SLAE::alloc_all(size_t n_size, size_t gg_size)
{
    n = n_size;
    ig = new size_t [n + 1];
    jg = new size_t [gg_size];
    di = new double [n];
    gg = new double [gg_size];
    rp = new double [n];
    x  = new double [n];

    memset(di, 0, sizeof(double) * n);
    memset(rp, 0, sizeof(double) * n);
    memset(x,  0, sizeof(double) * n);
    memset(gg, 0, sizeof(double) * gg_size);
}

void SLAE::add(size_t i, size_t j, double elem)
{
    if(i == j)
    {
        di[j] += elem;
        return;
    }

    if(j > i)
        swap(i, j);
    size_t ind = 0;
    bool flag = false;
    for(size_t k = ig[i]; k < ig[i + 1] && !flag; k++)
    {
        if(jg[k] == j)
        {
            ind = k;
            flag = true;
        }
    }
    gg[ind] += elem;
}

// ============================================================================

FEM::FEM()
{
    nodes = NULL;
    rects = NULL;
    phs   = NULL;
    pss   = NULL;

    nodes_num = 0;
    rects_num = 0;
    phs_num   = 0;
    pss_num   = 0;
}

FEM::~FEM()
{
    if(nodes) delete [] nodes;
    if(rects) delete [] rects;
    if(phs)   delete [] phs;
    if(pss)   delete [] pss;
}

void FEM::input()
{
    ifstream ifs;
    cout << "Reading data..." << endl;

    ifs.open("nodes.txt", ios::in);
    ifs >> nodes_num;
    cout << " > detected " << nodes_num << " nodes" << endl;
    nodes = new point [nodes_num];
    for(size_t i = 0; i < nodes_num; i++)
    {
        ifs >> nodes[i].x >> nodes[i].y;
        nodes[i].num = i;
    }
    ifs.close();

    ifs.open("phys.txt", ios::in);
    ifs >> phs_num;
    cout << " > detected " << phs_num << " physical areas" << endl;
    phs = new phys_area [phs_num];
    for(size_t i = 0; i < phs_num; i++)
    {
        ifs >> phs[i].sigma;
        phs[i].num = i;
    }
    ifs.close();

    ifs.open("fes.txt", ios::in);
    ifs >> rects_num;
    cout << " > detected " << rects_num << " rectangles" << endl;
    rects = new rectangle [rects_num];
    for(size_t i = 0; i < rects_num; i++)
    {
        size_t n;
        for(size_t j = 0; j < 4; j++)
        {
            ifs >> n;
            rects[i].nodes[j] = nodes + n;
        }
        //std::cout << "node 0" << *rects[i].nodes[0] << "\n";
        //std::cout << "node 1" << *rects[i].nodes[1] << "\n";
        //std::cout << "node 2" << *rects[i].nodes[2] << "\n";
        //std::cout << "node 3" << *rects[i].nodes[3] << "\n\n";

        ifs >> n;
        rects[i].ph = phs + n;
    }
    ifs.close();

    for(size_t i = 0; i < rects_num; i++)
        rects[i].init();

    ifs.open("sources.txt", ios::in);
    ifs >> pss_num;
    pss = new pair<point, double> [pss_num];
    for(size_t i = 0; i < pss_num; i++)
    {
        ifs >> pss[i].first.r >> pss[i].first.z >> pss[i].second;
    }
    ifs.close();
}

void FEM::make_portrait()
{
    // Формирование профиля (портрета)
    cout << "Generating profile..." << endl;
    size_t gg_size = 0;
    // Создаем массив списков для хранения связей
    set<size_t> * profile = new set<size_t> [nodes_num];

    // Связь есть, если узлы принадлежат одному КЭ
    // Поэтому обходим конечные элементы и добавляем в список общие вершины
    for(size_t i = 0; i < rects_num; i++)
        for(size_t j = 0; j < 4; j++)
            for(size_t k = 0; k < j; k++)
                profile[rects[i].nodes[k]->num].insert(rects[i].nodes[j]->num);

    // Удаляем повторяющиеся записи в спиках и сортируем их, попутно считая размер матрицы
    for(size_t i = 0; i < nodes_num; i++)
        gg_size += profile[i].size();
    slae.alloc_all(nodes_num, gg_size);

    // Заполнение профиля (портрета)
    slae.ig[0] = 0;
    slae.ig[1] = 0;
    size_t tmp = 0;
    for(size_t i = 0; i < slae.n; i++)
    {
        size_t k = 0;
        for(size_t j = 0; j <= i; j++)
        {
            // Если есть связь между i и j, значит в этом месте матрицы будет ненулевой элемент
            // занесем информацию об этом в jg
            if(profile[j].find(i) != profile[j].end())
            {
                slae.jg[tmp] = j;
                tmp++;
                k++;
            }
        }
        // а в ig занесем информацию о количестве ненулевых элементов в строке
        slae.ig[i + 1] = slae.ig[i] + k;
    }

    // Очистка списков
    for(size_t i = 0; i < nodes_num; i++)
        profile[i].clear();
    delete [] profile;
}

void FEM::assembling_global()
{
    memset(slae.di, 0, sizeof(double) * slae.n);
    memset(slae.rp, 0, sizeof(double) * slae.n);
    memset(slae.x,  0, sizeof(double) * slae.n);
    memset(slae.gg, 0, sizeof(double) * slae.ig[slae.n]);

    for(size_t k = 0; k < rects_num; k++)
    {
        for(size_t i = 0; i < 4; i++)
        {
            for(size_t j = 0; j <= i; j++)
            {
                slae.add(rects[k].nodes[i]->num, rects[k].nodes[j]->num,
                         rects[k].get_local_G(i, j));
            }

            /*
#warning "Here!"
            double rp = 0.0;
            for(size_t j = 0; j < 4; j++)
            {
                rp += rects[k].get_local_M(i, j) * -6.0;
            }
            slae.rp[rects[k].nodes[i]->num] += rp;
            */
        }
    }
}

void FEM::applying_bound()
{
    ifstream bound1_s;
    bound1_s.open("bound1.txt", ios::in);

    // Счетчик, сколько краевых применено
    size_t counter = 0;

    // Применение первых краевых условий
    while(bound1_s.good())
    {
        // Узел, в котором задано краевое
        size_t node_b1;
        // Значение краевого
        double value_b1;
        bound1_s >> node_b1 >> value_b1;
        if(!bound1_s.good())
            break;

        /*
#warning "Here!"
        value_b1 = nodes[node_b1].r * nodes[node_b1].r +
                nodes[node_b1].z * nodes[node_b1].z;
        */

        // Учет первых краевых "по-хорошему"
        // В диагональ пишем 1
        slae.di[node_b1] = 1.0;
        // В правую часть пишем значение краевого
        slae.rp[node_b1] = value_b1;
        // А вот тут все веселье
        // Нам надо занулить строку, а у нас симметричная матрица
        // Поэтому будем бегать по матрице, занулять стоки
        // А то, что было в столбцах - выкидывать в правую часть
        size_t i_s = slae.ig[node_b1], i_e = slae.ig[node_b1 + 1];
        for(size_t i = i_s; i < i_e; i++)
        {
            slae.rp[slae.jg[i]] -= slae.gg[i] * value_b1;
            slae.gg[i] = 0.0;
        }
        for(size_t p = node_b1 + 1; p < nodes_num; p++)
        {
            size_t i_s = slae.ig[p], i_e = slae.ig[p + 1];
            for(size_t i = i_s; i < i_e; i++)
            {
                if(slae.jg[i] == node_b1)
                {
                    slae.rp[p] -= slae.gg[i] * value_b1;
                    slae.gg[i] = 0.0;
                }
            }
        }

        // Увеличим счетчик
        counter++;
    }
    bound1_s.close();
    //cout << " > Applied " << counter << " boundary value of type I." << endl;
}

void FEM::applying_sources()
{
    for(size_t i = 0; i < pss_num; i++)
    {
        const rectangle * fe = get_fe(pss[i].first);
        for(size_t j = 0; j < 4; j++)
            slae.rp[fe->nodes[j]->num] += pss[i].second * fe->bfunc_2d(j, pss[i].first);
    }
    //cout << " > Applied " << pss_num << " point sources." << endl;
}

const rectangle * FEM::get_fe(const point & p) const
{
    for(size_t i = 0; i < rects_num; i++)
    {
        if(rects[i].inside(p))
            return rects + i;
    }
    cerr << "Warning: Target point " << p << " is outside of area" << endl;
    return NULL;
}

double FEM::get_solution(const point & p) const
{
    return get_solution(p, get_fe(p));
}

double FEM::get_solution(const point & p, const rectangle * fe) const
{
    if(fe)
    {
        double solution = 0.0;
        for(size_t i = 0; i < 4; i++)
            solution += fe->bfunc_2d(i, p) * slae.x[fe->nodes[i]->num];
        return solution;
    }
    return 0.0;
}

point FEM::get_grad(const point & p) const
{
    return get_grad(p, get_fe(p));
}

point FEM::get_grad(const point & p, const rectangle * fe) const
{
    if(fe)
    {
        point solution(0.0, 0.0);
        for(size_t i = 0; i < 4; i++)
        {
            double q = slae.x[fe->nodes[i]->num];
            point gbf = fe->grad_bfunc2d(i, p);
            solution.x += q * gbf.x;
            solution.y += q * gbf.y;
        }
        return solution;
    }
    return point(0.0, 0.0);
}

void FEM::out_tecplot(const string & tecplot_filename) const
{

    double min_x = 0, max_x = 4000, step_x = 200;
    double min_y = 0, max_y = 2000, step_y = 100;

    size_t nx = (size_t)((max_x - min_x) / step_x);
    size_t ny = (size_t)((max_y - min_y) / step_y);

    cout << "Writing to Tecplot ..." << endl;

    ofstream tecplot_file;
    tecplot_file.open(tecplot_filename.c_str(), ios::out);

    if(!tecplot_file.good())
    {
        cerr << "Error while writing file " << tecplot_filename << endl;
        return;
    }

    tecplot_file << "TITLE = \"Title must be here\"\n";
    tecplot_file << "VARIABLES = \"r\", \"z\", \"V\"\n";
    tecplot_file << "ZONE I= " << nx + 1 << ", J= " << ny + 1 << ", F=POINT\n";

    tecplot_file.precision(17);
    tecplot_file << scientific;

    for(size_t j = 0; j <= ny; j++)
    {
        double y = min_y + step_y * (double)j;
        for(size_t k = 0; k <= nx; k++)
        {
            double x = min_x + step_x * (double)k;

            point p = point(x, y);
            const rectangle * r = get_fe(p);

            double sol = get_solution(p, r);
            tecplot_file << x << " " << y << " " << sol << "\n";
        }
    }

    tecplot_file << "\n";
    tecplot_file.flush();
    tecplot_file.close();
}


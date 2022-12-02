#include "inverse.h"

void grid_generator()
{
    cout << "Generating grid..." << endl;
    /**/
        // Ключевые позиции по оси X
    const size_t num_x = 2;
    const double area_x_position[num_x] =
    {
        0.0,
        1000.0
    };
    // Ключевые позиции по оси Y
    const size_t num_y = 2;
    const double area_y_position[num_y] =
    {
        0.0,
        //100.0,
        //400.0,
        //500.0,
        1000.0
    };
    // Коэффициенты разрядки
    const double kx = 1.05;
    const double ky = 1.05;
    // Начальные шаги
    double hx = 2.0;
    double hy = 2.0;
    // Значения сигма в областях
    const size_t num_ph = num_y - 1;
    const double sigma[num_ph] =
    {
        0.3
        //0.1,
        //0.01,
        //0.1
    };
    // Точечные источники
    struct ps
    {
        union { double x; double r; };
        union { double y; double z; };
        double val;
    };
    const size_t num_ps = 2;
    ps pss[num_ps] =
    {
        {{1e-2}, {10.}, 10.0},
        {{1e-2}, {30.}, 10.0}
    };

    // КЭД
    //pss[0].val /= (2.0 * M_PI * pss[0].r);
    //pss[1].val /= -(2.0 * M_PI * pss[1].r);
    // ВЭЛ
    pss[0].val *= 1.0;
    pss[1].val *= - 1.0;

    // Генерация сетки по оси X
    vector<double> grid_x;
    // Номера узлов начала ключевых позиций
    vector<size_t> grid_x_pos;
    // Цикл по ключевым позициям
    for (size_t i = 0; i < num_x - 1; i++)
    {
        // Заносим первое значение в сетку
        grid_x.push_back(area_x_position[i]);
        // Запоминаем, где начало
        grid_x_pos.push_back(grid_x.size() - 1);
        // Получаем новое значение
        double new_pos = grid_x[grid_x.size() - 1] + hx;
        // Смотрим, не вышли ли за пределы области
        while ((new_pos < area_x_position[i + 1] && grid_x[grid_x.size() - 1] < area_x_position[i + 1]) ||
            (new_pos > area_x_position[i + 1] && grid_x[grid_x.size() - 1] > area_x_position[i + 1]))
        {
            // Если нет, заносим значение в сетку
            grid_x.push_back(new_pos);
            // Делаем разрядку
            hx *= kx;
            // Снова получаем новое значение
            new_pos += hx;
        }
        // На границе делать разрядку не стоит
        hx /= kx;
        // Если предпоследняя точка попала близко к границе
        if (fabs(grid_x[grid_x.size() - 1] - area_x_position[i + 1]) < fabs(hx) / 2.0 && grid_x[grid_x.size() - 1] != area_x_position[i])
        {
            // Отодвинем ее подальше
            grid_x[grid_x.size() - 1] = grid_x[grid_x.size() - 2] + (area_x_position[i + 1] - grid_x[grid_x.size() - 2]) / 2.0;
        }
    }
    // Поместим в сетку конец области
    grid_x.push_back(area_x_position[num_x - 1]);

    // Генерация сетки по оси Y
    vector<double> grid_y;
    // Номера узлов начала ключевых позиций
    vector<size_t> grid_y_pos;
    // Цикл по ключевым позициям
    for (size_t i = 0; i < num_y - 1; i++)
    {
        // Заносим первое значение в сетку
        grid_y.push_back(area_y_position[i]);
        // Запоминаем, где начало
        grid_y_pos.push_back(grid_y.size() - 1);
        // Получаем новое значение
        double new_pos = grid_y[grid_y.size() - 1] + hy;
        // Смотрим, не вышли ли за пределы области
        while ((new_pos < area_y_position[i + 1] && grid_y[grid_y.size() - 1] < area_y_position[i + 1]) ||
            (new_pos > area_y_position[i + 1] && grid_y[grid_y.size() - 1] > area_y_position[i + 1]))
        {
            // Если нет, заносим значение в сетку
            grid_y.push_back(new_pos);
            // Делаем разрядку
            hy *= ky;
            // Снова получаем новое значение
            new_pos += hy;
        }
        // На границе делать разрядку не стоит
        hy /= ky;
        // Если предпоследняя точка попала близко к границе
        if (fabs(grid_y[grid_y.size() - 1] - area_y_position[i + 1]) < fabs(hy) / 2.0 && grid_y[grid_y.size() - 1] != area_y_position[i])
        {
            // Отодвинем ее подальше
            grid_y[grid_y.size() - 1] = grid_y[grid_y.size() - 2] + (area_y_position[i + 1] - grid_y[grid_y.size() - 2]) / 2.0;
        }
    }
    // Поместим в сетку конец области
    grid_y.push_back(area_y_position[num_y - 1]);

    // Вывод в файл узлов
    ofstream nodes;
    nodes.open("nodes.txt", ios::out);
    nodes << grid_x.size() * grid_y.size() << endl;
    for (size_t j = 0; j < grid_y.size(); j++)
        for (size_t i = 0; i < grid_x.size(); i++)
            nodes << grid_x[i] << " \t" << grid_y[j] << endl;
    nodes.flush();
    nodes.close();

    // Вывод в файл физических областей
    ofstream ph_a;
    ph_a.open("phys.txt", ios::out);
    ph_a << num_ph << endl;
    for (size_t i = 0; i < num_ph; i++)
        ph_a << sigma[i] << endl;
    ph_a.flush();
    ph_a.close();

    // Вывод в файл всех конечных элементов
    ofstream finite_el;
    finite_el.open("fes.txt", ios::out);
    finite_el << (grid_x.size() - 1) * (grid_y.size() - 1) << endl;
    for (size_t j = 0; j < grid_y.size() - 1; j++)
    {
        for (size_t i = 0; i < grid_x.size() - 1; i++)
        {
            // Определение номера в глобальной нумерации
            size_t num[4];
            num[0] = grid_x.size() * j + i;
            num[1] = grid_x.size() * j + i + 1;
            num[2] = grid_x.size() * (j + 1) + i;
            num[3] = grid_x.size() * (j + 1) + i + 1;

            for (size_t k = 0; k < 4; k++)
                finite_el << num[k] << " \t";
            finite_el << endl;
            // Определение, к области с какой сигмой относится данный КЭ
            //if (j >= grid_y_pos[0] && j < grid_y_pos[1]) // Сигма1
            //    finite_el << 0 << endl;
            //if (j >= grid_y_pos[1] && j < grid_y_pos[2]) // Сигма2
            //    finite_el << 1 << endl;
            //if (j >= grid_y_pos[2] && j < grid_y_pos[3]) // Сигма3
            //    finite_el << 2 << endl;
            //if (j >= grid_y_pos[3]) // Сигма4
            //    finite_el << 3 << endl;
            
//#warning "Here!"
            finite_el << 0 << endl;
            

        }
    }
    finite_el.flush();
    finite_el.close();

    // Вывод в файл первых краевых условий
    ofstream bound1;
    bound1.open("bound1.txt", ios::out);
    for (size_t j = 1; j <= grid_y.size(); j++)
    {
        size_t num = grid_x.size() * j - 1;
        bound1 << num << " \t" << 0.0 << endl;
    }
    for (size_t i = 0; i < grid_x.size(); i++)
    {
        size_t num = grid_x.size() * (grid_y.size() - 1) + i;
        bound1 << num << " \t" << 0.0 << endl;
    }

    bound1.flush();
    bound1.close();

    // Вывод в файл точечных источников
    ofstream psss;
    psss.open("sources.txt", ios::out);
    psss << num_ps << endl;
    for (size_t i = 0; i < num_ps; i++)
        psss << pss[i].r << " \t"
        << pss[i].z << " \t"
        << pss[i].val << endl;
    psss.flush();
    psss.close();
}

int main()
{
    grid_generator();

    //inverse<parameter_sigma> inv;
    inverse<parameter_power> inv;
    inv.calc();

#if defined(_WIN32)
    system("pause");
#endif
    return 0;
}
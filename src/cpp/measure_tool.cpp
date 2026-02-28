#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

std::vector<std::pair<int, int>> bresenham_line(int y0, int x0, int y1, int x1) {
    std::vector<std::pair<int, int>> points;
    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    int x = x0, y = y0;  
    while (true) {
        points.push_back({y, x});
        if (x == x1 && y == y1) break;  

        int e2 = 2 * err;
        if (e2 > -dy) {  
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
    return points;
}



PYBIND11_MODULE(measure_tool, m) {
    m.doc() = "Measurement tools for fibre diameter measurement, including computation-intensive functions built in cpp";
    m.def("bresenham_line", &bresenham_line, "Return point coordinates set using Bresenham's algorithm");
}


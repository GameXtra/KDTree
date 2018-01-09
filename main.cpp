#include <CGAL/Gmpq.h>
#include <CGAL/Cartesian_d.h>
#include <boost/timer.hpp>
//#include <fstream>
//#include <sstream>
//#include <ostream>
//#include <set>
//#include <boost/lexical_cast.hpp>
//#include <CGAL/Kernel_d/Point_d.h>

#include "knn.h"


typedef CGAL::Gmpq Number_type;
typedef CGAL::Cartesian_d<Number_type> Kernel;
typedef Kernel::Point_d Point_d;

using namespace std;

Point_d random_point(size_t d) {
    std::vector<Number_type> point_to_be;
    for (int i = 0; i < d; ++i) {
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        Kernel::FT c = r;
        point_to_be.push_back(c);
    }
    Point_d point(d, point_to_be.begin(), point_to_be.end());
    return point;
}

void generate_statistics() {
    const size_t min_d = 1;
    const size_t max_d = 10;

    const size_t min_size = 10;
    const size_t max_size = 10000;
    const size_t size_scaling = 10;

    const size_t min_k = 1;
    const size_t k_scaling = 2;

    const int number_of_builds = 10;
    const int number_of_runs_per_k = 10;

    boost::timer timer;

    double total_build = 0;
    double total_run = 0;

    for (size_t d = min_d; d <= max_d; ++d) {
        cout << "dimensions: " << d << endl;
        for (size_t n = min_size; n <= max_size; n *= size_scaling) {
            cout << "       size: " << n << endl;
            vector<Point_d> points;
            for (int i = 0; i < n; ++i) points.push_back(random_point(d));
            double build_time = 0;
            Knn<Kernel> *knn = nullptr;
            for (int i = 0; i < number_of_builds; ++i) {
                delete knn;
                timer.restart();
                knn = new Knn<Kernel>(d, points.begin(), points.end());
                build_time += timer.elapsed();
                total_build += timer.elapsed();
            }

            build_time /= number_of_builds;
            cout << "               build time: " << build_time << endl;

            if (knn == nullptr) {
                cout << "Error: knn is null" << endl;
                return;
            }

            for (size_t k = min_k; k <= n; k *= k_scaling) {
                double search_time = 0;
                for (int runs = 0; runs < number_of_runs_per_k; ++runs) {
                    Point_d p = random_point(d);
                    std::vector<Point_d> res;
                    res.reserve(k);
                    timer.restart();
                    knn->find_points(k, p, std::back_inserter(res));
                    search_time += timer.elapsed();
                    total_run += timer.elapsed();
                }
                search_time /= number_of_runs_per_k;
                cout << "                       for k = " << k << ": " << search_time << endl;
            }

        }
    }
    cout << "total build " << total_build << endl;
    cout << "total run " << total_run << endl;
}

int main(int argc, char *argv[]) {
    generate_statistics();
    return 0;
}


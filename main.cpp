#include <fstream>
#include <sstream>
#include <ostream>
#include <set>
#include <boost/lexical_cast.hpp>
#include <boost/timer.hpp>
#include <CGAL/Cartesian_d.h>
#include <CGAL/Gmpq.h>
#include <CGAL/Kernel_d/Point_d.h>


#include "knn.h"


typedef CGAL::Gmpq                      Number_type;
typedef CGAL::Cartesian_d<Number_type>  Kernel;
typedef Kernel::Point_d                 Point_d;

Point_d read_point(size_t d, std::ifstream& is)
{

    std::vector<Number_type> point_to_be;
    for(int i=0;i<d;++i)
    {
        Kernel::FT c;
        is >> c;
        point_to_be.push_back(c);
    }
    Point_d point(d,point_to_be.begin(),point_to_be.end());
    return point;
}

Kernel::FT squared_dist_d(Point_d& p, Point_d& q, size_t d)
{
    Kernel::FT res = 0;
    for (size_t i=0; i < d ;++i)
    {
        res += (p[i] - q[i]) * (p[i] - q[i]);
    }
    return res;
}


bool lexic_smaller(const Point_d& p, const Point_d&q, size_t d)
{
    for (size_t i=0; i < d; ++i)
    {
        if (p[i]< q[i])
            return true;
        if (p[i] > q[i])
            return false;
    }
    return false;
}

struct lex_compare {
    size_t d_;
    lex_compare(size_t d): d_(d) {}

    bool operator() (const Point_d& lhs, const Point_d& rhs) const {
        return lexic_smaller(lhs, rhs, d_);
    }
};


using namespace std;

//argv: ./a.out dimension input_points_file query_points_file
int main(int argc, char* argv[])
{
    size_t d = 3;

    const auto* build_filename ="/home/ido/KDTree/1000_points_3d.txt";
    std::ifstream  build_is;
    build_is.open(build_filename);
    if (!build_is.is_open()) {
        std::cerr << "Failed to open " << build_filename << "!" << std::endl;
        return -1;
    }
    size_t bulidn;
    build_is >>  bulidn;
    std::vector<Point_d> build_points;
    for (size_t i = 0; i <  bulidn; ++i) {
        build_points.push_back(read_point(d, build_is));
    }

    const auto* test_filename = "/home/ido/KDTree/100_points_3d.txt";
    std::ifstream test_is;
    test_is.open(test_filename);
    if (!test_is.is_open()) {
        std::cerr << "Failed to open " << test_filename << "!" << std::endl;
        return -1;
    }
    size_t testn;
    test_is >> testn;
    std::vector<Point_d> test_points;
    for (auto i = 0; i <  testn; ++i) {
        test_points.push_back(read_point(d, test_is));
    }

    //create a set of the build points
    std::set<Point_d, lex_compare> pset(build_points.begin(), build_points.end(), lex_compare(d));


    Knn<Kernel> knn(d, build_points.begin(), build_points.end());
    for (size_t k = 2; k < 10; ++k) {
        for (auto it = test_points.begin(); it != test_points.end(); ++it) {
            std::vector<Point_d> res;
            res.reserve(k);
            boost::timer timer;
            knn.find_points(k,*it,std::back_inserter(res));
            auto secs = timer.elapsed();
            std::cout<<secs<<" time"<<std::endl;



            //----------------------------------------------//
            //code for testing the output of knn.find_points
            //----------------------------------------------//

            //1. we should get k points or less if k > n
            if ((k!= res.size()  && k<=build_points.size()) ||
                res.size() > k)
            {
                std::cerr << "Not enough neighbors" << std::endl;
                return -1;
            }

            //2. make sure that all reported points are points
            //    from the build_points
            for (auto rit = res.begin(); rit != res.end(); ++rit)
            {
                if (pset.find(*rit)  == pset.end())
                {
                    std::cerr << "Reported point was not part of the build point set" << std::endl;
                    return -1;
                }
            }

            //3. make sure that each reported point appears once in res
            std::set<Point_d, lex_compare> res_set(res.begin(), res.end(), lex_compare(d));
            if (res_set.size() < res.size())
            {
                std::cerr << "A point was reported as a neighbor more than once" << std::endl;
                return -1;

            }


            //4. make sure that there is no point from build_points that is
            //   closer to the query point than the farthest point in res
            //   and that is not reported in res.
            //   Make sure that every point whose distance to the query is as the
            //   distance of the farthest point to the query, is lexicographically greater than the farthest point in res.

            //find maximal in res (point whose dist to test_point it is maximal)

            size_t ind_of_max = 0;
            Kernel::FT maxdist = squared_dist_d(res[0], *it, d);
            for (auto  j=1; j < res.size() ; ++j){
                Kernel::FT j_dist = squared_dist_d(res[j], *it, d);
                if (maxdist < j_dist){
                    ind_of_max = j;
                    maxdist = j_dist;
                }
                else if (maxdist ==  j_dist){
                    //if (res[ind_of_max] < res[j])
                    if (lexic_smaller(res[ind_of_max], res[j], d))
                        ind_of_max = j;
                }
            }


            Point_d& max_neighbor = res[ind_of_max];

            //verify that no other point in the data set is closer to *it then the farthest in res.
            //If there is a point whose distance from *it is the same then verify it is lexicographically greater than the point in res obtaining the max distance
            for (auto pit = build_points.begin(); pit != build_points.end(); ++pit)
            {
                if (*pit == max_neighbor)
                    continue;
                Kernel::FT pit_dist = squared_dist_d(*pit, *it,d);
                //if ((pit_dist == maxdist && *pit < max_neighbor) || (pit_dist < maxdist))
                if ((pit_dist == maxdist && lexic_smaller(*pit,max_neighbor, d)) || (pit_dist < maxdist))
                {
                    //verify that *pit is reported in res
                    if (res_set.find(*pit) == res_set.end())
                    {
                        std::cerr << "A potential neighbor was not reported" << std::endl;
                        return -1;
                    }
                }
            }

        }
    }

    return 0;
}


//#include <CGAL/Gmpq.h>
//#include <CGAL/Cartesian_d.h>
//#include <boost/timer.hpp>
////#include <fstream>
////#include <sstream>
////#include <ostream>
////#include <set>
////#include <boost/lexical_cast.hpp>
////#include <CGAL/Kernel_d/Point_d.h>
//
//#include "knn.h"
//
//
//typedef CGAL::Gmpq Number_type;
//typedef CGAL::Cartesian_d<Number_type> Kernel;
//typedef Kernel::Point_d Point_d;
//
//using namespace std;
//
//Point_d random_point(size_t d) {
//    std::vector<Number_type> point_to_be;
//    for (int i = 0; i < d; ++i) {
//        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
//        Kernel::FT c = r;
//        point_to_be.push_back(c);
//    }
//    Point_d point(d, point_to_be.begin(), point_to_be.end());
//    return point;
//}
//
//void generate_statistics() {
//    const size_t min_d = 1;
//    const size_t max_d = 10;
//
//    const size_t min_size = 10;
//    const size_t max_size = 10000;
//    const size_t size_scaling = 10;
//
//    const size_t min_k = 1;
//    const size_t k_scaling = 2;
//
//    const int number_of_builds = 10;
//    const int number_of_runs_per_k = 10;
//
//    boost::timer timer;
//
//    double total_build = 0;
//    double total_run = 0;
//
//    for (size_t d = min_d; d <= max_d; ++d) {
//        cout << "dimensions: " << d << endl;
//        for (size_t n = min_size; n <= max_size; n *= size_scaling) {
//            cout << "       size: " << n << endl;
//            vector<Point_d> points;
//            for (int i = 0; i < n; ++i) points.push_back(random_point(d));
//            double build_time = 0;
//            Knn<Kernel> *knn = nullptr;
//            for (int i = 0; i < number_of_builds; ++i) {
//                delete knn;
//                timer.restart();
//                knn = new Knn<Kernel>(d, points.begin(), points.end());
//                build_time += timer.elapsed();
//                total_build += timer.elapsed();
//            }
//
//            build_time /= number_of_builds;
//            cout << "               build time: " << build_time << endl;
//
//            if (knn == nullptr) {
//                cout << "Error: knn is null" << endl;
//                return;
//            }
//
//            for (size_t k = min_k; k <= n; k *= k_scaling) {
//                double search_time = 0;
//                for (int runs = 0; runs < number_of_runs_per_k; ++runs) {
//                    Point_d p = random_point(d);
//                    std::vector<Point_d> res;
//                    res.reserve(k);
//                    timer.restart();
//                    knn->find_points(k, p, std::back_inserter(res));
//                    search_time += timer.elapsed();
//                    total_run += timer.elapsed();
//                }
//                search_time /= number_of_runs_per_k;
//                cout << "                       for k = " << k << ": " << search_time << endl;
//            }
//
//        }
//    }
//    cout << "total build " << total_build << endl;
//    cout << "total run " << total_run << endl;
//}
//
//int main(int argc, char *argv[]) {
//    generate_statistics();
//    return 0;
//}
//

#include "knnNode.h"
#include "knnRepository.h"

using namespace MyKnn;
using namespace std;

template<typename Kernel>
class Knn {
public:
    typedef typename Kernel::Point_d Point_d;
    typedef typename Kernel::FT FT;

    // input: d - the dimension
    // iterator to the input points
    template<typename InputIterator>
    Knn(size_t d, InputIterator beginPoints, InputIterator endPoints): d(d) {
        for (int i = 1; beginPoints != endPoints; ++beginPoints, ++i)
            this->points.push_back(*beginPoints);
        Point_d ***points_index_by_axis = sort_by_each_axis(d);
        this->root = unique_ptr<KnnNode<Kernel>>(
                new KnnNode<Kernel>(points_index_by_axis, points.size(), 0, d));
        for (int i = 0; i < d; ++i)
            delete[] points_index_by_axis[i];
        delete[] points_index_by_axis;
    }


    //input:   const reference to a d dimensional vector which represent a d-point.
    //output:  a vector of the indexes of the k-nearest-neighbors points todo : indexes or point_d??
    template<typename OutputIterator>
    OutputIterator find_points(size_t k, const Point_d &it, OutputIterator oi) {
        KnnRepository<Kernel> knnRepository(*root, it, k, d);
        for (auto point: knnRepository.get_results())
            oi++ = *point;
        return oi;
    }

private:
    unique_ptr <KnnNode<Kernel>> root;
    vector <Point_d> points;
    size_t d;

    Point_d ***sort_by_each_axis(size_t d) {
        auto ***res = new Point_d **[d];
        for (auto i = 0; i < d; ++i) {
            res[i] = new Point_d *[points.size()];
            for (auto j = 0; j < points.size(); ++j)
                res[i][j] = &(points[j]);
            sort(res[i], res[i] + points.size(), [&](const Point_d *a, const Point_d *b) -> bool {
                return (*a)[i] < (*b)[i];
            });
        }
        return res;
    }
};

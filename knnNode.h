//
// Created by t-idkess on 10-Jan-18.
//

#ifndef EXECUTABLE_KNNNODE_H
#define EXECUTABLE_KNNNODE_H

#include <memory>
#include <set>
#include <vector>
#include <cstring>
#include "constants.h"

using namespace std;

namespace MyKnn {
    template<typename Kernel>
    class KnnNode {
        typedef typename Kernel::Point_d Point_d;
        typedef typename Kernel::FT FT;

    public:
        KnnNode(Point_d ***points_sorted_by_axis, size_t number_of_points, size_t index_to_sort_by, const size_t d) : d(
                d) {
            if (number_of_points == 0) return; // is empty.

            for (int i = 0; i < number_of_points; ++i) this->points.push_back(points_sorted_by_axis[0][i]);
            update_min_max(points_sorted_by_axis, number_of_points);
            if (number_of_points <= MAX_NUMBER_OF_POINTS_IN_NODE) return; // no need to split.

            size_t m = number_of_points / 2;
            Point_d ***left_points_ordered_by_axis, ***right_points_ordered_by_axis;
            this->create_splited_sorted_by_axis_array(points_sorted_by_axis, &left_points_ordered_by_axis,
                                                      &right_points_ordered_by_axis, number_of_points,
                                                      index_to_sort_by);
            this->left = unique_ptr<KnnNode>(
                    new KnnNode(left_points_ordered_by_axis, m, (index_to_sort_by + 1) % d, d));
            this->right = unique_ptr<KnnNode>(
                    new KnnNode(right_points_ordered_by_axis, number_of_points - m, (index_to_sort_by + 1) % d, d));

            delete[] left_points_ordered_by_axis, delete[] right_points_ordered_by_axis;
        }

        // todo make this faster by caching the last result.
        Point_d get_closest_point_possible(const Point_d &point) const {
            vector<FT> point_to_be;
            for (int i = 0; i < d; i++) {
                if (min_point[i] > point[i]) {
                    point_to_be.push_back(min_point[i]);
                } else if (max_point[i] < point[i]) {
                    point_to_be.push_back(max_point[i]);
                } else {
                    point_to_be.push_back(point[i]);
                }
            }
            return Point_d(d, point_to_be.begin(), point_to_be.end());;
        }

        size_t size() const {
            return points.size();
        }

        const vector<const Point_d *> &get_points() const {
            return points;
        }

        KnnNode &get_left() const {
            return *left;
        }

        KnnNode &get_right() const {
            return *right;
        }

    private:
        vector<const Point_d *> points;
        unique_ptr<KnnNode> left, right;
        Point_d min_point;
        Point_d max_point;
        const size_t d;

        void update_min_max(Point_d ***points_ordered_by_axis, size_t number_of_points) {
            vector<FT> point_to_be_min;
            vector<FT> point_to_be_max;
            for (int i = 0; i < d; ++i) {
                point_to_be_min.push_back((*(points_ordered_by_axis[i][0]))[i]);
                point_to_be_max.push_back((*(points_ordered_by_axis[i][number_of_points - 1]))[i]);
            }
            min_point = Point_d(d, point_to_be_min.begin(), point_to_be_min.end());
            max_point = Point_d(d, point_to_be_max.begin(), point_to_be_max.end());
        }

        void create_splited_sorted_by_axis_array(Point_d ***origin, Point_d ****pleft, Point_d ****pright, size_t n,
                                                 size_t index_to_sort_by) {
            auto **temp = new Point_d *[n];
            size_t m = n / 2;
            set<Point_d *> leftSet;
            for (int i = 0; i < m; ++i)
                leftSet.insert(origin[index_to_sort_by][i]);
            (*pleft) = new Point_d **[d], (*pright) = new Point_d **[d];
            for (int i = 0; i < d; ++i) {
                int left_index = 0, right_index = (int) m;
                for (int j = 0; j < n; ++j) {
                    if (leftSet.find(origin[i][j]) != leftSet.end()) {
                        temp[left_index++] = origin[i][j];
                    } else {
                        temp[right_index++] = origin[i][j];
                    }
                }
                memcpy(&(origin[i][0]), &(temp[0]), n * sizeof(Point_d *));
                (*pleft)[i] = &(origin[i][0]), (*pright)[i] = &(origin[i][m]);
            }
            delete[] temp;
        }
    };
}
#endif //EXECUTABLE_KNNNODE_H

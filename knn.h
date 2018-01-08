#include <vector>
#include <queue>

#define MAX_NUMBER_OF_POINTS_IN_NODE 3

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
        auto points_index_by_axis = sort_by_each_axis(d);
        this->root = unique_ptr<KnnNode>(
                new KnnNode(points_index_by_axis, points.size(), 0, d));
        for (int i = 0; i < d; ++i)
            delete[] points_index_by_axis[i];
        delete[] points_index_by_axis;
    }


    //input:   const reference to a d dimensional vector which represent a d-point.
    //output:  a vector of the indexes of the k-nearest-neighbors points todo : indexes or point_d??
    template<typename OutputIterator>
    OutputIterator find_points(size_t k, const Point_d &it, OutputIterator oi) {
        KNearestPointRepository knnRepository(*root, it, k, d);
        for (auto point: knnRepository.get_results())
            oi++ = *point;
        return oi;
    }

private:

    class KnnNode {
    public:
        KnnNode(Point_d ***points_sorted_by_axis,
                size_t number_of_points, size_t index_to_sort_by,
                const size_t d) : d(d) {
            if (number_of_points == 0)
                return; // is empty.
            for (int i = 0; i < number_of_points; ++i)
                this->points.push_back(points_sorted_by_axis[0][i]);
            update_min_max(points_sorted_by_axis, number_of_points);
            if (number_of_points <= MAX_NUMBER_OF_POINTS_IN_NODE)
                return; // no need to split.
            size_t m = number_of_points / 2;
            Point_d ***left_points_ordered_by_axis, ***right_points_ordered_by_axis;
            this->create_splited_sorted_by_axis_array(points_sorted_by_axis,
                                                      &left_points_ordered_by_axis,
                                                      &right_points_ordered_by_axis,
                                                      number_of_points, index_to_sort_by);
            this->left = unique_ptr<KnnNode>(new KnnNode(left_points_ordered_by_axis, m,
                                                         (index_to_sort_by + 1) % d, d));
            this->right = unique_ptr<KnnNode>(new KnnNode(right_points_ordered_by_axis,
                                                          number_of_points - m,
                                                          (index_to_sort_by + 1) % d, d));
            for (int i = 0; i < d; i++)
                delete[] left_points_ordered_by_axis[i], delete[] right_points_ordered_by_axis[i];
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
            size_t m = n / 2;
            set<Point_d *> leftSet;
            for (int i = 0; i < m; ++i)
                leftSet.insert(origin[index_to_sort_by][i]);
            (*pleft) = new Point_d **[d], (*pright) = new Point_d **[d];
            for (int i = 0; i < d; ++i) {
                (*pleft)[i] = new Point_d *[m], (*pright)[i] = new Point_d *[n - m];
                int left_index = 0, right_index = 0;
                for (int j = 0; j < n; ++j) {
                    if (leftSet.find(origin[i][j]) != leftSet.end())
                        (*pleft)[i][left_index++] = origin[i][j];
                    else
                        (*pright)[i][right_index++] = origin[i][j];
                }
            }
        }
    };

    typedef typename Knn<Kernel>::KnnNode KnnNode;

    unique_ptr<KnnNode> root;
    vector<Point_d> points;
    size_t d;

    Point_d ***sort_by_each_axis(size_t d) {
        Point_d ***res = new Point_d **[d];
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

    class KNearestPointRepository {
    private:
        // todo use another type maybe that can be iterated without pop... cost a lot.
        typedef priority_queue<pair<const Point_d *, FT>, vector<pair<const Point_d *, FT>>,
                typename Knn<Kernel>::CompareDistLess> points_priority_queue_t;
        typedef priority_queue<pair<KnnNode *, FT>, vector<pair<KnnNode *, FT>>,
                typename Knn<Kernel>::CompareDistGreater> nodes_priority_queue_t;
    public :
        KNearestPointRepository(KnnNode &root,
                                const Point_d &current_query_point,
                                size_t k, size_t d) : current_query_point(current_query_point),
                                                      k(k), d(d) {
            points_priority_queue_t knnPointQueue;
            nodes_priority_queue_t knnNodeQueue;

            this->add_KddNode_to_queue(knnNodeQueue, root);

            while (knnNodeQueue.size()) {
                pair<KnnNode *, FT> currentNode = knnNodeQueue.top();
                knnNodeQueue.pop();
                if (currentNode.first->size() <= k - knnPointQueue.size() ||
                    currentNode.first->size() <= MAX_NUMBER_OF_POINTS_IN_NODE) {
                    for (auto point : currentNode.first->get_points()) {
                        FT distance = squared_dist_d(*point, current_query_point, d);
                        knnPointQueue.push(make_pair(point, distance));
                        while (knnPointQueue.size() > k) {
                            knnPointQueue.pop();
                        }
                    }
                } else {
                    if (knnPointQueue.size() == k && currentNode.second > knnPointQueue.top().second) {
                        // if queue is full and the distance to the most far away
                        // point is smaller, then there's no need to continue;
                        break;
                    } else {
                        this->add_KddNode_to_queue(knnNodeQueue, (currentNode.first->get_left()));
                        this->add_KddNode_to_queue(knnNodeQueue, (currentNode.first->get_right()));
                    }
                }
            }
            while (knnPointQueue.size()) {
                result.push_back(knnPointQueue.top().first);
                knnPointQueue.pop();
            }
        }

        const vector<const Point_d *> &get_results() const { return result; }

    private:
        const Point_d &current_query_point;
        const size_t k, d;
        vector<const Point_d *> result;

        template<typename queue_t>
        void add_KddNode_to_queue(queue_t &queue, KnnNode &node) {
            Point_d closestPoint = node.get_closest_point_possible(current_query_point);
            FT distance = squared_dist_d(closestPoint, current_query_point, d);
            queue.emplace(&node, distance);
        }


    };

public:
    class CompareDistLess {
    public:
        bool operator()(pair<const Point_d *, FT> &a, pair<const Point_d *, FT> &b) {
            if (a.second == b.second)
                return lexic_smaller(*(a.first), *(b.first), (size_t) ((a.first)->dimension()));
            return a.second < b.second;
        }
    };

    class CompareDistGreater {
    public:
        bool operator()(pair<KnnNode *, FT> &a,
                        pair<KnnNode *, FT> &b) {
            return a.second > b.second;
        }
    };

    static bool lexic_smaller(const Point_d &p, const Point_d &q, size_t d) {
        for (size_t i = 0; i < d; ++i)
            if (p[i] < q[i])
                return true;
            else if (p[i] > q[i])
                return false;
        return false;
    }

    static FT squared_dist_d(const Point_d &p, const Point_d &q, size_t d) {
        FT res = 0;
        for (size_t i = 0; i < d; ++i) {
            FT r = (p[i] - q[i]);
            res += r * r;
        }
        return res;
    }
};

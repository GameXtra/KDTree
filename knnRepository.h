//
// Created by t-idkess on 10-Jan-18.
//

#ifndef EXECUTABLE_KNNREPOSITORY_H
#define EXECUTABLE_KNNREPOSITORY_H

#include "constants.h"


namespace MyKnn {
    template<typename Kernel>
    class KnnRepository {
    private:
        // todo use another type maybe that can be iterated without pop... cost a lot.
        typedef priority_queue<pair<const Point_d *, FT>, vector<pair<const Point_d *, FT>>,
                typename Knn<Kernel>::CompareDistLess> points_priority_queue_t;
        typedef priority_queue <pair<KnnNode *, FT>, vector<pair < KnnNode * , FT>>,
        typename Knn<Kernel>::CompareDistGreater>
        nodes_priority_queue_t;
    public :
        KNearestPointRepository(KnnNode
        &root,
        const Point_d &current_query_point,
                size_t
        k,
        size_t d
        ) :

        current_query_point (current_query_point),
        k(k), d(d) {
            points_priority_queue_t knnPointQueue;
            nodes_priority_queue_t knnNodeQueue;

            this->add_KddNode_to_queue(knnNodeQueue, root);

            while (knnNodeQueue.size()) {
                pair < KnnNode * , FT > currentNode = knnNodeQueue.top();
                knnNodeQueue.pop();
                if (currentNode.first->size() <= k - knnPointQueue.size() ||
                    currentNode.first->size() <= MAX_NUMBER_OF_POINTS_IN_NODE) {
                    for (auto point : currentNode.first->get_points()) {
                        FT distance = squared_dist_d(*point, current_query_point, d);
                        if (knnPointQueue.size() == k) {
                            if (distance > knnPointQueue.top().second) continue;
                            if (distance != knnPointQueue.top().second ||
                                lexic_smaller(*point, *(knnPointQueue.top().first))) {
                                knnPointQueue.push(make_pair(point, distance));
                                knnPointQueue.pop();
                            }
                        } else {
                            knnPointQueue.push(make_pair(point, distance));
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


    class CompareDistLess {
    public:
        bool operator()(pair<const Point_d *, FT> &a, pair<const Point_d *, FT> &b) {
            if (a.second == b.second)
                return lexic_smaller(*(a.first), *(b.first), (size_t)((a.first)->dimension()));
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
}


#endif //EXECUTABLE_KNNREPOSITORY_H

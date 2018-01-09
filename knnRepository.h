//
// Created by t-idkess on 10-Jan-18.
//

#ifndef EXECUTABLE_KNNREPOSITORY_H
#define EXECUTABLE_KNNREPOSITORY_H

#include <queue>
#include "constants.h"
#include "knnNode.h"


namespace MyKnn {
    template<typename Kernel>
    class KnnRepository {
        typedef typename Kernel::Point_d Point_d;
        typedef typename Kernel::FT FT;

    public :
        KnnRepository(KnnNode <Kernel> &root, const Point_d &current_query_point, size_t k, size_t d)
                : current_query_point(current_query_point), k(k),
                  d(d) {
            // todo use another type maybe that can be iterated without pop... cost a lot.
            priority_queue <pair<const Point_d *, FT>, vector<pair<const Point_d *, FT>>, CompareDistLess>
                    knnPointQueue;
            priority_queue <pair<KnnNode<Kernel> *, FT>, vector<pair<KnnNode<Kernel> *, FT>>, CompareDistGreater>
                    knnNodeQueue;

            this->add_KddNode_to_queue(knnNodeQueue, root);

            while (knnNodeQueue.size()) {
                pair<KnnNode<Kernel> *, FT> currentNode = knnNodeQueue.top();
                knnNodeQueue.pop();
                if (currentNode.first->size() <= k - knnPointQueue.size() ||
                    currentNode.first->size() <= MAX_NUMBER_OF_POINTS_IN_NODE) {
                    for (auto point : currentNode.first->get_points()) {
                        FT distance = squared_dist_d(*point, current_query_point, d);
                        if (knnPointQueue.size() == k) {
                            if (distance > knnPointQueue.top().second) continue;
                            if (distance != knnPointQueue.top().second ||
                                lexic_smaller(*point, *(knnPointQueue.top().first), d)) {
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
            bool operator()(pair<KnnNode<Kernel> *, FT> &a,
                            pair<KnnNode<Kernel> *, FT> &b) {
                return a.second > b.second;
            }
        };

    private:
        const Point_d &current_query_point;
        const size_t k, d;
        vector<const Point_d *> result;

        template<typename queue_t>
        void add_KddNode_to_queue(queue_t &queue, KnnNode<Kernel> &node) {
            Point_d closestPoint = node.get_closest_point_possible(current_query_point);
            FT distance = squared_dist_d(closestPoint, current_query_point, d);
            queue.emplace(&node, distance);
        }

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


}


#endif //EXECUTABLE_KNNREPOSITORY_H

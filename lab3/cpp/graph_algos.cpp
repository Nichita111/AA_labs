#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

struct GraphInstance {
    int n;
    std::vector<std::pair<int, int>> edges;
    bool directed;
    bool weighted;
};

static std::uint64_t edge_key(int u, int v) {
    return (static_cast<std::uint64_t>(u) << 32) | static_cast<std::uint32_t>(v);
}

static int max_edges_simple(int n, bool directed) {
    if (directed) {
        return n * (n - 1);
    }
    return n * (n - 1) / 2;
}

static int edge_target(int n, bool directed, const std::string &density, int max_edges) {
    if (density == "sparse") {
        return std::min(max_edges, std::max(1, 2 * n));
    }
    return std::min(max_edges, std::max(1, static_cast<int>(0.6 * max_edges)));
}

static std::vector<std::pair<int, int>> random_edges_simple(
    int n,
    int m,
    bool directed,
    std::mt19937 &rng
) {
    std::unordered_set<std::uint64_t> edges;
    edges.reserve(static_cast<size_t>(m) * 2);
    std::uniform_int_distribution<int> dist(0, n - 1);

    while (static_cast<int>(edges.size()) < m) {
        int u = dist(rng);
        int v = dist(rng);
        if (u == v) {
            continue;
        }
        if (!directed && u > v) {
            std::swap(u, v);
        }
        edges.insert(edge_key(u, v));
    }

    std::vector<std::pair<int, int>> out;
    out.reserve(edges.size());
    for (std::uint64_t k : edges) {
        int u = static_cast<int>(k >> 32);
        int v = static_cast<int>(k & 0xFFFFFFFFu);
        out.emplace_back(u, v);
    }
    return out;
}

static std::vector<std::pair<int, int>> add_random_edges_existing(
    int n,
    const std::vector<std::pair<int, int>> &edges,
    bool directed,
    int target_m,
    std::mt19937 &rng
) {
    std::unordered_set<std::uint64_t> edge_set;
    edge_set.reserve(static_cast<size_t>(target_m) * 2);
    for (const auto &e : edges) {
        int u = e.first;
        int v = e.second;
        if (!directed && u > v) {
            std::swap(u, v);
        }
        edge_set.insert(edge_key(u, v));
    }

    std::uniform_int_distribution<int> dist(0, n - 1);
    while (static_cast<int>(edge_set.size()) < target_m) {
        int u = dist(rng);
        int v = dist(rng);
        if (u == v) {
            continue;
        }
        if (!directed && u > v) {
            std::swap(u, v);
        }
        edge_set.insert(edge_key(u, v));
    }

    std::vector<std::pair<int, int>> out;
    out.reserve(edge_set.size());
    for (std::uint64_t k : edge_set) {
        int u = static_cast<int>(k >> 32);
        int v = static_cast<int>(k & 0xFFFFFFFFu);
        out.emplace_back(u, v);
    }
    return out;
}

static std::vector<std::pair<int, int>> grid_edges(int n) {
    int side = std::max(2, static_cast<int>(std::sqrt(n)));
    int rows = side;
    int cols = std::max(2, (n + rows - 1) / rows);
    std::vector<std::pair<int, int>> edges;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int u = r * cols + c;
            if (u >= n) {
                continue;
            }
            if (c + 1 < cols && u + 1 < n) {
                edges.emplace_back(u, u + 1);
            }
            if (r + 1 < rows) {
                int v = (r + 1) * cols + c;
                if (v < n) {
                    edges.emplace_back(u, v);
                }
            }
        }
    }
    return edges;
}

static std::vector<std::pair<int, int>> cycle_edges(int n) {
    std::vector<std::pair<int, int>> edges;
    edges.reserve(n);
    for (int i = 0; i < n; ++i) {
        edges.emplace_back(i, (i + 1) % n);
    }
    return edges;
}

static std::vector<std::pair<int, int>> tree_edges(int n, std::mt19937 &rng) {
    std::vector<std::pair<int, int>> edges;
    edges.reserve(n > 1 ? n - 1 : 0);
    for (int v = 1; v < n; ++v) {
        std::uniform_int_distribution<int> dist(0, v - 1);
        int p = dist(rng);
        edges.emplace_back(p, v);
    }
    return edges;
}

static std::vector<std::pair<int, int>> wheel_edges(int n) {
    if (n < 4) {
        return cycle_edges(n);
    }
    int hub = n - 1;
    std::vector<std::pair<int, int>> edges;
    edges.reserve((n - 1) * 2);
    for (int i = 0; i < n - 1; ++i) {
        edges.emplace_back(i, (i + 1) % (n - 1));
        edges.emplace_back(hub, i);
    }
    return edges;
}

static std::vector<std::pair<int, int>> bipartite_edges(
    int n,
    int m,
    std::mt19937 &rng
) {
    int left = n / 2;
    int right = n - left;
    std::unordered_set<std::uint64_t> edges;
    edges.reserve(static_cast<size_t>(m) * 2);

    std::uniform_int_distribution<int> dist_left(0, left - 1);
    std::uniform_int_distribution<int> dist_right(0, right - 1);

    while (static_cast<int>(edges.size()) < m) {
        int u = dist_left(rng);
        int v = left + dist_right(rng);
        int a = std::min(u, v);
        int b = std::max(u, v);
        edges.insert(edge_key(a, b));
    }

    std::vector<std::pair<int, int>> out;
    out.reserve(edges.size());
    for (std::uint64_t k : edges) {
        int u = static_cast<int>(k >> 32);
        int v = static_cast<int>(k & 0xFFFFFFFFu);
        out.emplace_back(u, v);
    }
    return out;
}

static std::vector<std::pair<int, int>> regular_graph_edges(
    int n,
    int degree,
    std::mt19937 &rng
) {
    if (degree >= n) {
        throw std::invalid_argument("degree must be < n");
    }
    if ((n * degree) % 2 != 0) {
        throw std::invalid_argument("n * degree must be even");
    }

    for (int attempt = 0; attempt < 50; ++attempt) {
        std::vector<int> stubs;
        stubs.reserve(n * degree);
        for (int v = 0; v < n; ++v) {
            for (int k = 0; k < degree; ++k) {
                stubs.push_back(v);
            }
        }
        std::shuffle(stubs.begin(), stubs.end(), rng);

        std::unordered_set<std::uint64_t> edges;
        edges.reserve(static_cast<size_t>(n * degree));
        bool valid = true;

        while (!stubs.empty()) {
            int u = stubs.back();
            stubs.pop_back();
            int v = stubs.back();
            stubs.pop_back();
            if (u == v) {
                valid = false;
                break;
            }
            int a = std::min(u, v);
            int b = std::max(u, v);
            std::uint64_t key = edge_key(a, b);
            if (edges.find(key) != edges.end()) {
                valid = false;
                break;
            }
            edges.insert(key);
        }

        if (valid) {
            std::vector<std::pair<int, int>> out;
            out.reserve(edges.size());
            for (std::uint64_t k : edges) {
                int u = static_cast<int>(k >> 32);
                int v = static_cast<int>(k & 0xFFFFFFFFu);
                out.emplace_back(u, v);
            }
            return out;
        }
    }

    int target_m = std::min(max_edges_simple(n, false), (n * degree) / 2);
    return random_edges_simple(n, target_m, false, rng);
}

static GraphInstance generate_graph(
    const std::string &graph_type,
    int n,
    const std::string &density,
    std::mt19937 &rng
) {
    bool directed = false;
    bool weighted = false;
    std::vector<std::pair<int, int>> edges;

    if (graph_type == "Directed graph" || graph_type == "Directed Multigraph") {
        directed = true;
    }
    if (graph_type == "Weighted graph") {
        weighted = true;
    }

    int max_edges = max_edges_simple(n, directed);
    int target_m = edge_target(n, directed, density, max_edges);

    if (graph_type == "Simple graph" || graph_type == "Multigraph" || graph_type == "Pseudograph") {
        edges = random_edges_simple(n, target_m, directed, rng);
    } else if (graph_type == "Directed graph" || graph_type == "Directed Multigraph") {
        edges = random_edges_simple(n, target_m, true, rng);
    } else if (graph_type == "Complete graph") {
        if (density == "dense") {
            edges = random_edges_simple(n, max_edges, directed, rng);
        } else {
            edges = random_edges_simple(n, target_m, directed, rng);
        }
    } else if (graph_type == "Wheels") {
        edges = wheel_edges(n);
        if (density == "dense") {
            edges = add_random_edges_existing(n, edges, directed, target_m, rng);
        }
    } else if (graph_type == "Bipartite graph") {
        int max_bip = (n / 2) * (n - n / 2);
        if (density == "dense") {
            target_m = std::min(max_bip, std::max(1, static_cast<int>(0.6 * max_bip)));
        } else {
            target_m = std::min(max_bip, std::max(1, 2 * n));
        }
        edges = bipartite_edges(n, target_m, rng);
    } else if (graph_type == "Regular graph") {
        int degree = density == "dense" ? std::max(2, std::min(n - 1, n / 2)) : (n > 2 ? 2 : 1);
        if ((n * degree) % 2 != 0) {
            degree = std::max(1, degree - 1);
        }
        edges = regular_graph_edges(n, degree, rng);
    } else if (graph_type == "Planar graph (grid-based)") {
        edges = grid_edges(n);
        if (density == "dense") {
            edges = add_random_edges_existing(n, edges, directed, target_m, rng);
        }
    } else if (graph_type == "Trees") {
        edges = tree_edges(n, rng);
        if (density == "dense") {
            edges = add_random_edges_existing(n, edges, directed, target_m, rng);
        }
    } else if (graph_type == "Cyclic graph") {
        edges = cycle_edges(n);
        if (density == "dense") {
            edges = add_random_edges_existing(n, edges, directed, target_m, rng);
        }
    } else if (graph_type == "Weighted graph") {
        edges = random_edges_simple(n, target_m, directed, rng);
    } else {
        throw std::invalid_argument("Unknown graph type: " + graph_type);
    }

    return GraphInstance{n, edges, directed, weighted};
}

static std::vector<std::vector<int>> build_adj_list(
    int n,
    const std::vector<std::pair<int, int>> &edges,
    bool directed
) {
    std::vector<std::vector<int>> adj(n);
    for (const auto &e : edges) {
        adj[e.first].push_back(e.second);
        if (!directed) {
            adj[e.second].push_back(e.first);
        }
    }
    for (auto &row : adj) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    return adj;
}

static std::vector<std::vector<int>> order_adj_by_degree(
    const std::vector<std::vector<int>> &adj
) {
    int n = static_cast<int>(adj.size());
    std::vector<int> degree(n, 0);
    for (int i = 0; i < n; ++i) {
        degree[i] = static_cast<int>(adj[i].size());
    }

    std::vector<std::vector<int>> ordered = adj;
    for (int i = 0; i < n; ++i) {
        auto &row = ordered[i];
        std::stable_sort(row.begin(), row.end(), [&](int a, int b) {
            if (degree[a] != degree[b]) {
                return degree[a] > degree[b];
            }
            return a < b;
        });
    }
    return ordered;
}

struct CSRMatrix {
    int n;
    std::vector<int> row;
    std::vector<int> col;

    CSRMatrix(int n_, const std::vector<std::pair<int, int>> &edges, bool directed) : n(n_) {
        std::vector<std::vector<int>> adj(n);
        adj.reserve(n);
        for (const auto &e : edges) {
            adj[e.first].push_back(e.second);
            if (!directed && e.first != e.second) {
                adj[e.second].push_back(e.first);
            }
        }
        for (auto &row_list : adj) {
            std::sort(row_list.begin(), row_list.end());
            row_list.erase(std::unique(row_list.begin(), row_list.end()), row_list.end());
        }

        row.assign(n + 1, 0);
        for (int i = 0; i < n; ++i) {
            row[i] = static_cast<int>(col.size());
            for (int v : adj[i]) {
                col.push_back(v);
            }
        }
        row[n] = static_cast<int>(col.size());
    }
};

static CSRMatrix build_csr_from_adj(const std::vector<std::vector<int>> &adj) {
    CSRMatrix csr(0, {}, false);
    int n = static_cast<int>(adj.size());
    csr.n = n;
    csr.row.assign(n + 1, 0);
    csr.col.clear();
    csr.col.reserve([&]() {
        size_t total = 0;
        for (const auto &row : adj) {
            total += row.size();
        }
        return total;
    }());

    for (int i = 0; i < n; ++i) {
        csr.row[i] = static_cast<int>(csr.col.size());
        for (int v : adj[i]) {
            csr.col.push_back(v);
        }
    }
    csr.row[n] = static_cast<int>(csr.col.size());
    return csr;
}

static int classical_dfs(const std::vector<std::vector<int>> &adj, int source) {
    int n = static_cast<int>(adj.size());
    std::vector<char> visited(n, 0);
    std::vector<int> stack;
    stack.reserve(n);
    stack.push_back(source);
    int count = 0;

    while (!stack.empty()) {
        int u = stack.back();
        stack.pop_back();
        if (visited[u]) {
            continue;
        }
        visited[u] = 1;
        ++count;
        const auto &nbrs = adj[u];
        for (auto it = nbrs.rbegin(); it != nbrs.rend(); ++it) {
            int v = *it;
            if (!visited[v]) {
                stack.push_back(v);
            }
        }
    }
    return count;
}

static int classical_bfs(const std::vector<std::vector<int>> &adj, int source) {
    int n = static_cast<int>(adj.size());
    std::vector<char> visited(n, 0);
    std::deque<int> q;
    q.push_back(source);
    visited[source] = 1;
    int count = 1;

    while (!q.empty()) {
        int u = q.front();
        q.pop_front();
        for (int v : adj[u]) {
            if (!visited[v]) {
                visited[v] = 1;
                ++count;
                q.push_back(v);
            }
        }
    }
    return count;
}

static int optimized_dfs_iterative(const std::vector<std::vector<int>> &adj, int source) {
    int n = static_cast<int>(adj.size());
    static thread_local std::vector<char> visited;
    static thread_local std::vector<int> stack;
    visited.assign(n, 0);
    stack.clear();
    stack.reserve(n);

    visited[source] = 1;
    stack.push_back(source);
    int count = 0;

    while (!stack.empty()) {
        int u = stack.back();
        stack.pop_back();
        ++count;
        const auto &nbrs = adj[u];
        for (auto it = nbrs.rbegin(); it != nbrs.rend(); ++it) {
            int v = *it;
            if (!visited[v]) {
                visited[v] = 1;
                stack.push_back(v);
            }
        }
    }
    return count;
}

static int optimized_dfs_csr(const CSRMatrix &csr, int source) {
    int n = csr.n;
    static thread_local std::vector<char> visited;
    static thread_local std::vector<int> stack;
    visited.assign(n, 0);
    stack.clear();
    stack.reserve(n);

    visited[source] = 1;
    stack.push_back(source);
    int count = 0;

    while (!stack.empty()) {
        int u = stack.back();
        stack.pop_back();
        ++count;
        for (int idx = csr.row[u]; idx < csr.row[u + 1]; ++idx) {
            int v = csr.col[idx];
            if (!visited[v]) {
                visited[v] = 1;
                stack.push_back(v);
            }
        }
    }
    return count;
}

static int optimized_bfs_csr(const CSRMatrix &csr, int source) {
    int n = csr.n;
    static thread_local std::vector<char> visited;
    static thread_local std::vector<char> x;
    static thread_local std::vector<char> y;
    static thread_local std::vector<int> L;
    visited.assign(n, 0);
    x.assign(n, 0);
    y.assign(n, 0);
    L.assign(n, 0);

    x[source] = 1;
    visited[source] = 1;
    L[0] = source;
    int start = 0;
    int end = 1;
    int z = 1;
    int count = 1;

    while (start < end) {
        for (int qi = start; qi < end; ++qi) {
            int j = L[qi];
            for (int idx = csr.row[j]; idx < csr.row[j + 1]; ++idx) {
                int i = csr.col[idx];
                if (!visited[i]) {
                    y[i] = x[j] | y[i];
                    visited[i] = 1;
                    L[z] = i;
                    ++z;
                    ++count;
                }
            }
            visited[j] = 1;
            x[j] = 0;
        }
        start = end;
        end = z;
        x.swap(y);
        std::fill(y.begin(), y.end(), 0);
    }

    return count;
}

static double time_call(const std::function<void()> &fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

static py::dict run_benchmark(
    const std::string &graph_type,
    int n,
    const std::string &density,
    int trials,
    int seed
) {
    std::vector<double> classical_dfs_times;
    std::vector<double> classical_bfs_times;
    std::vector<double> optimized_dfs_times;
    std::vector<double> optimized_bfs_times;
    classical_dfs_times.reserve(trials);
    classical_bfs_times.reserve(trials);
    optimized_dfs_times.reserve(trials);
    optimized_bfs_times.reserve(trials);

    for (int t = 0; t < trials; ++t) {
        std::mt19937 rng(static_cast<std::uint32_t>(seed + t * 101 + n * 17));
        GraphInstance g = generate_graph(graph_type, n, density, rng);
        auto adj = build_adj_list(g.n, g.edges, g.directed);
        auto adj_ordered = order_adj_by_degree(adj);
        CSRMatrix csr(g.n, g.edges, g.directed);
        CSRMatrix csr_opt = build_csr_from_adj(adj_ordered);

        classical_dfs_times.push_back(time_call([&]() { classical_dfs(adj, 0); }));
        classical_bfs_times.push_back(time_call([&]() { classical_bfs(adj, 0); }));
        optimized_dfs_times.push_back(time_call([&]() { optimized_dfs_csr(csr_opt, 0); }));
        optimized_bfs_times.push_back(time_call([&]() { optimized_bfs_csr(csr, 0); }));
    }

    auto avg = [](const std::vector<double> &vals) {
        double sum = 0.0;
        for (double v : vals) {
            sum += v;
        }
        return vals.empty() ? 0.0 : sum / static_cast<double>(vals.size());
    };

    py::dict result;
    result["classical_dfs"] = avg(classical_dfs_times);
    result["classical_bfs"] = avg(classical_bfs_times);
    result["optimized_dfs"] = avg(optimized_dfs_times);
    result["optimized_bfs"] = avg(optimized_bfs_times);
    return result;
}

static py::dict run_all(
    const std::vector<std::string> &graph_types,
    const std::vector<std::string> &densities,
    const std::vector<int> &sizes,
    int trials,
    int seed
) {
    py::dict results;

    for (const auto &graph_type : graph_types) {
        for (const auto &density : densities) {
            py::dict size_map;
            for (int n : sizes) {
                py::dict avg_times = run_benchmark(graph_type, n, density, trials, seed);
                size_map[py::int_(n)] = avg_times;
            }
            py::tuple key = py::make_tuple(graph_type, density);
            results[key] = size_map;
        }
    }

    return results;
}

PYBIND11_MODULE(graph_algos_cpp, m) {
    m.doc() = "C++ graph traversal benchmarks (pybind11)";
    m.def("run_benchmark", &run_benchmark, py::arg("graph_type"), py::arg("n"), py::arg("density"), py::arg("trials"), py::arg("seed"));
    m.def("run_all", &run_all, py::arg("graph_types"), py::arg("densities"), py::arg("sizes"), py::arg("trials"), py::arg("seed"));
}

[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a set of undirected links (edges) to connect all listed sites (nodes) into a minimum-cost spanning tree, using a single-commodity flow formulation to ensure connectivity, with explicit binary edge selection and flow variables, and constraints as described.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree (MST) model with single-commodity flow connectivity enforcement.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: set of all sites from `network_nodes.csv` (denote as N).
    - Edges: set of all undirected links from `network_edges.csv` (denote as E).
    - Directed arcs: for each undirected edge (i, j) in E, both (i, j) and (j, i) are considered as possible flow arcs (denote as A).
4.  **Define Decision Variables:**
    - `y_ij` = 1 if undirected edge (i, j) is selected in the spanning tree, 0 otherwise. Type: GRB.BINARY.
    - `f_ij` = amount of flow sent from the root along directed arc (i, j). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    - Construction costs for each edge from `network_edges.csv` column 'ConstructionCost'.
    - Node list from `network_nodes.csv` column 'Node' (used to define N and count n).
    - Root node from `network_parameters.csv` row where 'Parameter' = 'RootNode', value in 'Value'.
    - The number of nodes n = total rows in `network_nodes.csv`.
    - For flow-to-edge linking, M = n - 1.
6.  **Formulate Objective:** Minimize the total construction cost of selected edges: sum over all undirected edges (i, j) of `ConstructionCost[i,j] * y_ij`.
7.  **Formulate Constraints:**
    - Constraint 1 (Edge Count): The number of selected edges is exactly n - 1: sum over all (i, j) in E of y_ij = n - 1.
    - Constraint 2 (Root Flow Balance): At the root node r, the total flow sent out equals n - 1: sum over all arcs (r, j) in A of f_rj = n - 1.
    - Constraint 3 (Non-root Flow Balance): For each non-root node k in N \ {r}, the net flow in minus flow out equals 1: sum over all arcs (i, k) in A of f_ik minus sum over all arcs (k, j) in A of f_kj = 1.
    - Constraint 4 (Flow-to-Edge Linking): For each directed arc (i, j) in A, flow on arc cannot exceed M times edge selection: f_ij ≤ (n - 1) * y_ij, where y_ij refers to the undirected edge {i, j}.
    - Constraint 5 (Nonnegativity): All flow variables f_ij ≥ 0.
    - Constraint 6 (Binary): All edge selection variables y_ij ∈ {0, 1}.
[Abstract Model Plan END]
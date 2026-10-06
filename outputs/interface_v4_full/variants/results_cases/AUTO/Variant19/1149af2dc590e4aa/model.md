[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost spanning tree model for a network of sites, using a single-commodity flow formulation. The model must select a set of undirected edges to connect all nodes (sites) at minimum total construction cost, using binary edge-selection variables and directed flow variables to ensure connectivity from a designated root node.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree (MST) with single-commodity flow connectivity enforcement.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: All rows from `network_nodes.csv` (set N, 9 nodes: V1–V9).
    - Undirected Edges: All rows from `network_edges.csv` (set E, 16 edges, each between Node1 and Node2).
    - Directed Arcs: For each undirected edge (i, j), both (i, j) and (j, i) are considered as possible flow arcs.
4.  **Define Decision Variables:**
    -   `y_ij` = 1 if undirected edge between nodes i and j is selected in the spanning tree, 0 otherwise. Type: GRB.BINARY. (One variable per undirected edge in E.)
    -   `f_ij` = amount of flow sent from the root node along directed arc (i, j). Type: GRB.CONTINUOUS (nonnegative). (One variable per directed arc corresponding to each undirected edge in both directions.)
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Construction costs from the 'ConstructionCost' column in `network_edges.csv` (for each undirected edge).
    -   Node set: All nodes from `network_nodes.csv` (column 'Node').
    -   Edge set: All undirected edges from `network_edges.csv` (columns 'Node1', 'Node2').
    -   Root node: Value from `network_parameters.csv` where Parameter = 'RootNode' (e.g., V1).
    -   M parameter: Set to n-1, where n = number of nodes (here, 9).
6.  **Formulate Objective:** Minimize the total construction cost of selected edges: sum over all undirected edges (i, j) of ConstructionCost[i, j] * y_ij.
7.  **Formulate Constraints:**
    -   Constraint 1 (Edge Count): The number of selected edges must be exactly n-1: sum over all (i, j) in E of y_ij = n-1.
    -   Constraint 2 (Root Node Flow Balance): The root node (from network_parameters.csv) must send out exactly n-1 units of flow: sum over all arcs (root, j) of f_root,j - sum over all arcs (j, root) of f_j,root = n-1.
    -   Constraint 3 (Non-root Node Flow Balance): For every non-root node k, net inflow minus outflow must be 1: sum over all arcs (j, k) of f_jk - sum over all arcs (k, j) of f_kj = 1.
    -   Constraint 4 (Flow-to-Edge Linking): For every directed arc (i, j), the flow on arc (i, j) cannot exceed M times the selection variable for the corresponding undirected edge: f_ij ≤ (n-1) * y_ij, and similarly for f_ji ≤ (n-1) * y_ij.
    -   Constraint 5 (Nonnegativity): All flow variables f_ij ≥ 0.
    -   Constraint 6 (Binary): All edge selection variables y_ij ∈ {0, 1}.
[Abstract Model Plan END]
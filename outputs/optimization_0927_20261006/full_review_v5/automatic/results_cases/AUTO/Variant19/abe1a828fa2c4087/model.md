[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost spanning tree model for a network, ensuring all sites (nodes) are connected via selected undirected edges, using a single-commodity flow formulation rooted at a specified node. The model must minimize construction cost, select exactly n-1 edges, enforce flow-based connectivity, and link flow to edge selection.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree with flow-based connectivity constraints.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: set of all nodes from `network_nodes.csv`.
    - Undirected Edges: set of all unordered node pairs (i, j) from `network_edges.csv`.
    - Directed Arcs: for each undirected edge (i, j), both (i, j) and (j, i) as possible flow directions.
4.  **Define Decision Variables:**
    - `y_ij` = 1 if undirected edge (i, j) is selected in the spanning tree; 0 otherwise. Type: GRB.BINARY.
    - `f_ij` = amount of connectivity flow sent from the root along directed arc (i, j). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    - Construction costs for each undirected edge from `network_edges.csv` column 'ConstructionCost'.
    - Node set from `network_nodes.csv` column 'Node'.
    - Root node from `network_parameters.csv` where 'Parameter' = 'RootNode'.
    - Number of nodes n = total rows in `network_nodes.csv`.
    - Edge-arc mapping: each undirected edge (i, j) gives rise to two directed arcs (i, j) and (j, i).
    - M = n-1 (maximum possible flow sent from root).
6.  **Formulate Objective:** Minimize the total construction cost of selected edges: sum over all undirected edges (i, j) of `ConstructionCost[i,j] * y_ij`.
7.  **Formulate Constraints:**
    - Constraint 1 (Edge Count): The total number of selected edges is exactly n-1: sum over all (i, j) of y_ij = n-1.
    - Constraint 2 (Root Flow Balance): At the root node, total flow sent out equals n-1: sum over all arcs (root, j) of f_root,j = n-1.
    - Constraint 3 (Non-root Flow Balance): For each non-root node k, net flow in minus flow out equals 1: sum over all arcs (i, k) of f_ik minus sum over all arcs (k, j) of f_kj = 1.
    - Constraint 4 (Flow-to-Edge Linking): For each directed arc (i, j), flow f_ij <= (n-1) * y_ij, ensuring flow can only use selected edges.
    - Constraint 5 (Nonnegativity): All flow variables f_ij >= 0.
    - Constraint 6 (Binary Edge Selection): All y_ij are binary (0 or 1).
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost spanning tree model for a network of sites, using a single-commodity flow formulation. The model must select a set of undirected edges to connect all nodes (sites) at minimum construction cost, using binary edge-selection variables and directed flow variables to ensure connectivity from a specified root node.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem for the Minimum Spanning Tree (MST) using a single-commodity flow formulation.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: set of all site nodes from `network_nodes.csv`.
    - Undirected Edges: set of all available undirected links from `network_edges.csv`.
    - Directed Arcs: for each undirected edge (i, j), both (i, j) and (j, i) are considered as directed arcs.
4.  **Define Decision Variables:**
    - `y_ij` = 1 if undirected edge between nodes i and j is selected in the spanning tree; 0 otherwise. Type: GRB.BINARY.
    - `f_ij` = amount of flow sent from the root node along directed arc (i, j). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    - Construction costs for each undirected edge from the 'ConstructionCost' column in `network_edges.csv`.
    - Node set from the 'Node' column in `network_nodes.csv`.
    - Root node from the 'Value' column in `network_parameters.csv` where 'Parameter' is 'RootNode'.
    - The number of nodes, n, is the total number of rows in `network_nodes.csv`.
6.  **Formulate Objective:** Minimize the total construction cost of selected edges: sum over all undirected edges of (ConstructionCost * y_ij).
7.  **Formulate Constraints:**
    - Constraint 1 (Edge Count): The total number of selected edges is exactly n-1, i.e., sum over all y_ij = n-1.
    - Constraint 2 (Root Node Flow Balance): For the root node, the total flow sent out equals n-1 (sum over all arcs (root, j) of f_root,j = n-1).
    - Constraint 3 (Non-Root Node Flow Balance): For each non-root node k, the total flow in minus the total flow out equals 1 (sum over all arcs (i, k) of f_ik - sum over all arcs (k, j) of f_kj = 1).
    - Constraint 4 (Flow-to-Edge Linking): For each directed arc (i, j), the flow f_ij is less than or equal to (n-1) * y_ij, ensuring flow can only be sent along selected edges.
    - Constraint 5 (Nonnegativity): All flow variables f_ij are nonnegative.
    - Constraint 6 (Binary Edge Selection): All y_ij variables are binary (0 or 1).
[Abstract Model Plan END]
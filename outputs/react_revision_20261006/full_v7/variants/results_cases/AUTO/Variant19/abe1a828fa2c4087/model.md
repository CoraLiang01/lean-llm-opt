[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost spanning tree model for a network of sites, ensuring all nodes are connected via selected undirected edges, using a single-commodity flow formulation rooted at a specified node. The model must select exactly n-1 edges, enforce connectivity via flow constraints, and minimize total construction cost.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem for the Minimum Spanning Tree (MST) using a single-commodity flow formulation.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: All rows from network_nodes.csv (set N, with n = 9 nodes: V1, V2, ..., V9).
    - Undirected Edges: All rows from network_edges.csv (set E, 16 undirected edges, each between Node1 and Node2).
    - Directed Arcs: For each undirected edge (i, j), both (i, j) and (j, i) are considered as possible flow directions (set A).
4.  **Define Decision Variables:**
    -   `y_ij` = 1 if undirected edge between nodes i and j is selected in the spanning tree, 0 otherwise. Type: GRB.BINARY. (Defined for all (i, j) in E, with i < j to avoid duplicates.)
    -   `f_ij` = amount of flow sent from the root node through directed arc (i, j). Type: GRB.CONTINUOUS, lower bound 0. (Defined for all directed arcs (i, j) in A.)
5.  **Identify Parameters (from Schema):**
    -   Construction costs for each undirected edge from network_edges.csv column 'ConstructionCost'.
    -   Node set from network_nodes.csv column 'Node'.
    -   Root node from network_parameters.csv row where Parameter = 'RootNode', Value = e.g., 'V1'.
    -   The number of nodes n = 9 (from network_nodes.csv).
    -   M = n - 1 = 8 (used in flow-to-edge linking constraints).
6.  **Formulate Objective:** Minimize the total construction cost of selected edges: sum over all undirected edges (i, j) of ConstructionCost[i, j] * y_ij.
7.  **Formulate Constraints:**
    -   Constraint 1 (Edge Count): The number of selected edges must be exactly n-1: sum over all (i, j) in E of y_ij = n - 1.
    -   Constraint 2 (Root Node Flow Balance): The total flow sent out from the root node (e.g., V1) must be exactly n-1: sum over all arcs (root, j) of f_root,j - sum over all arcs (j, root) of f_j,root = n - 1.
    -   Constraint 3 (Non-root Node Flow Balance): For each non-root node k, the net flow in minus flow out must be exactly 1: sum over all arcs (i, k) of f_ik - sum over all arcs (k, j) of f_kj = 1, for all k ≠ root.
    -   Constraint 4 (Flow-to-Edge Linking): For each directed arc (i, j), the flow on arc (i, j) cannot exceed M times the selection variable for the corresponding undirected edge: f_ij ≤ M * y_{min(i,j), max(i,j)} (i.e., flow can only occur on selected edges, regardless of direction).
    -   Constraint 5 (Nonnegativity): All flow variables f_ij ≥ 0.
    -   Constraint 6 (Binary Edge Selection): All y_ij ∈ {0, 1}.
[Abstract Model Plan END]
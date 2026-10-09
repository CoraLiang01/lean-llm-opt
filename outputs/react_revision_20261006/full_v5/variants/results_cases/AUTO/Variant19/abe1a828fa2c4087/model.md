[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost spanning tree model for a network of sites, ensuring all nodes are connected via selected undirected edges, using a single-commodity flow formulation rooted at a specified node. The model must select exactly n-1 edges, enforce connectivity via flow variables, and minimize total construction cost.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem for the Minimum Spanning Tree (MST) using a single-commodity flow formulation.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: All rows from network_nodes.csv (set N, with n = 9 nodes: V1, V2, ..., V9).
    - Undirected Edges: All rows from network_edges.csv (set E, 16 undirected edges, each between Node1 and Node2).
    - Directed Arcs: For each undirected edge (i, j), both (i, j) and (j, i) are considered as possible flow directions (set A).
4.  **Define Decision Variables:**
    -   `y_ij` = 1 if undirected edge between nodes i and j is selected in the spanning tree, 0 otherwise. Type: GRB.BINARY. (One variable per undirected edge in E.)
    -   `f_ij` = amount of connectivity flow sent from the root node through directed arc (i, j). Type: GRB.CONTINUOUS (nonnegative). (One variable per directed arc in A.)
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'ConstructionCost' from network_edges.csv, associated with each undirected edge (i, j).
    -   Node set: 'Node' from network_nodes.csv (for defining N and identifying the root).
    -   Edge set: 'Node1', 'Node2' from network_edges.csv (for defining E and A).
    -   Root node: 'Value' in network_parameters.csv where 'Parameter' == 'RootNode' (e.g., V1).
    -   M parameter: M = n-1 = 8 (since n = 9).
6.  **Formulate Objective:** Minimize the total construction cost of selected edges: sum over all undirected edges (i, j) of ConstructionCost[i, j] * y_ij.
7.  **Formulate Constraints:**
    -   Constraint 1 (Edge Count): The number of selected edges must be exactly n-1: sum over all (i, j) in E of y_ij = n-1.
    -   Constraint 2 (Root Node Flow Balance): At the root node (e.g., V1), the total flow sent out equals n-1: sum over all arcs (root, j) of f_root,j = n-1.
    -   Constraint 3 (Non-root Node Flow Balance): For each non-root node k, the net flow in minus flow out equals 1: sum over all arcs (i, k) of f_ik - sum over all arcs (k, j) of f_kj = 1.
    -   Constraint 4 (Flow-to-Edge Linking): For each directed arc (i, j), the flow on arc (i, j) cannot exceed M times the selection variable for the corresponding undirected edge: f_ij ≤ M * y_ij, where y_ij refers to the undirected edge {i, j}.
    -   Constraint 5 (Nonnegativity): All flow variables f_ij ≥ 0.
    -   Constraint 6 (Binary Edge Selection): All y_ij ∈ {0, 1}.
[Abstract Model Plan END]
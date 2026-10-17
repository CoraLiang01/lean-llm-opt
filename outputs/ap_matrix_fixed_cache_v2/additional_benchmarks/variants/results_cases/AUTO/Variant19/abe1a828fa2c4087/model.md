[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost spanning tree model for a network of sites, using a single-commodity flow formulation. The model must select a set of undirected edges to connect all nodes (sites) at minimum total construction cost, with flow variables ensuring connectivity from a specified root node. The model must include edge selection, flow-balance, and linking constraints, using the provided node, edge, and parameter data.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem for the Minimum Spanning Tree (MST) using a single-commodity flow formulation.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: set of all site nodes from `network_nodes.csv`.
    - Edges: set of all undirected edges from `network_edges.csv`.
    - Directed arcs: for each undirected edge, both possible directions (i→j and j→i).
4.  **Define Decision Variables:**
    - `y_ij` = 1 if undirected edge between nodes i and j is selected in the spanning tree; 0 otherwise. Type: GRB.BINARY.
    - `f_ij` = amount of flow sent from the root node along directed arc (i, j). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    - Construction costs for each undirected edge from the 'ConstructionCost' column in `network_edges.csv`.
    - Node set from the 'Node' column in `network_nodes.csv`.
    - Root node from the 'Value' field in `network_parameters.csv` where 'Parameter' is 'RootNode'.
    - Number of nodes, n, is the count of rows in `network_nodes.csv`.
6.  **Formulate Objective:** Minimize the total construction cost of selected edges: sum over all undirected edges of (ConstructionCost * y_ij).
7.  **Formulate Constraints:**
    - Constraint 1 (Edge Count): The number of selected edges must be exactly n-1, where n is the number of nodes.
    - Constraint 2 (Root Flow Balance): The root node must send out exactly n-1 units of flow (one unit to each other node).
    - Constraint 3 (Non-root Flow Balance): Each non-root node must have net inflow of exactly 1 unit (sum of incoming flows minus outgoing flows equals 1).
    - Constraint 4 (Flow-to-Edge Linking): For each directed arc (i, j), the flow f_ij cannot exceed (n-1) times y_ij, ensuring flow only on selected edges.
    - Constraint 5 (Nonnegativity): All flow variables f_ij must be greater than or equal to zero.
    - Constraint 6 (Binary Edge Selection): All y_ij variables must be binary (0 or 1).
[Abstract Model Plan END]
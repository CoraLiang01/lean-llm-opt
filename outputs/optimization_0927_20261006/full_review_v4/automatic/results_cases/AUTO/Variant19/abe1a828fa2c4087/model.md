[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost spanning tree model for a network of sites, ensuring all nodes are connected via selected undirected edges, using a single-commodity flow formulation rooted at a specified node. The model must minimize total construction cost, select exactly n-1 edges, enforce flow-based connectivity, and link flow to edge selection.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree (MST) with single-commodity flow connectivity constraints.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: set of all site nodes from `network_nodes.csv`.
    - Edges: set of all available undirected edges from `network_edges.csv`.
    - Directed arcs: for each undirected edge, both possible directions (i→j and j→i).
4.  **Define Decision Variables:**
    - `y_ij` = 1 if undirected edge between nodes i and j is selected in the spanning tree; 0 otherwise. Type: GRB.BINARY.
    - `f_ij` = amount of single-commodity flow sent from the root node along directed arc (i, j). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    - Construction costs for each undirected edge from the 'ConstructionCost' column in `network_edges.csv`.
    - Node set from the 'Node' column in `network_nodes.csv`.
    - Root node from the 'Value' column in `network_parameters.csv` where 'Parameter' is 'RootNode'.
    - Number of nodes n = total rows in `network_nodes.csv`.
    - For flow-to-edge linking, M = n-1 (as specified in the query).
6.  **Formulate Objective:** Minimize the total construction cost, i.e., the sum over all undirected edges of (ConstructionCost × y_ij).
7.  **Formulate Constraints:**
    - Constraint 1 (Edge Count): The total number of selected edges is exactly n-1, i.e., sum over all y_ij = n-1.
    - Constraint 2 (Root Node Flow Balance): At the root node, the total flow sent out equals n-1 (sum of f_root,j over all outgoing arcs from the root = n-1).
    - Constraint 3 (Non-root Node Flow Balance): For each non-root node, the net flow in minus flow out equals 1 (sum of f_i,j into node minus sum of f_j,i out of node = 1 for each non-root node).
    - Constraint 4 (Flow-to-Edge Linking): For each directed arc (i, j), the flow f_ij is less than or equal to (n-1) × y_ij, ensuring flow can only be sent along selected edges.
    - Constraint 5 (Nonnegativity): All flow variables f_ij are nonnegative.
    - Constraint 6 (Binary Edge Selection): All y_ij variables are binary (0 or 1).
[Abstract Model Plan END]
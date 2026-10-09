[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a minimum-cost communication backbone (minimum spanning tree) connecting all substations (nodes), using a subset of available undirected links (edges), with connectivity enforced via a flow-based formulation rooted at a specified node. The model must select exactly n-1 links, ensure all nodes are connected, and use auxiliary flow variables to guarantee connectivity.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree (MST) with flow-based connectivity constraints.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: set of substations, from all rows in `network_nodes.csv` (column 'Node').
    - Edges: set of candidate undirected links, from all rows in `network_edges.csv` (columns 'Node1', 'Node2').
    - For flow variables: ordered pairs (i, j) for both directions of each undirected edge.
4.  **Define Decision Variables:**
    -   `y_e` = 1 if undirected link e (between nodes i and j) is built, 0 otherwise. Type: GRB.BINARY.
    -   `f_ij` = amount of auxiliary flow sent from root node to node j via directed arc (i, j) (for each direction of each candidate edge). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Construction costs for each edge: from `network_edges.csv` column 'ConstructionCost'.
    -   Node list: from `network_nodes.csv` column 'Node'.
    -   Edge list: from `network_edges.csv` columns 'Node1', 'Node2'.
    -   Root node: from `network_parameters.csv` row where 'Parameter' == 'RootNode', value in 'Value'.
    -   Number of nodes n: count of rows in `network_nodes.csv`.
6.  **Formulate Objective:** Minimize total construction cost, i.e., sum over all candidate edges of (ConstructionCost * y_e), where y_e indicates if edge e is built.
7.  **Formulate Constraints:**
    -   Constraint 1 (Spanning Tree Size): The total number of selected links is exactly n-1, i.e., sum over all y_e = n-1.
    -   Constraint 2 (Flow Conservation for Connectivity): For each node k ≠ root, the net inflow of auxiliary flow at node k is exactly 1 (i.e., sum of incoming f_ik minus sum of outgoing f_ki = 1). For the root node, the net outflow is n-1 (i.e., sum of outgoing f_root,j minus sum of incoming f_j,root = n-1). For all other nodes, net flow is zero.
    -   Constraint 3 (Flow-Edge Linking): For each directed arc (i, j) corresponding to undirected edge e, the auxiliary flow f_ij ≤ (n-1) * y_e. This ensures flow can only pass on selected edges.
    -   Constraint 4 (Nonnegativity): All flow variables f_ij ≥ 0.
    -   Constraint 5 (Binary Restrictions): All link selection variables y_e ∈ {0,1}.
[Abstract Model Plan END]
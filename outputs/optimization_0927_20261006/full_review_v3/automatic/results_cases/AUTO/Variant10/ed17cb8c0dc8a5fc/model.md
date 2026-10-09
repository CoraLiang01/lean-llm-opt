[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a minimum-cost communication backbone that connects all substations (nodes) using a subset of available undirected links, ensuring network connectivity via a minimum spanning tree. Connectivity is enforced using a flow-based formulation with a designated root node, and the model must select exactly n-1 links, where n is the number of nodes.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree (MST) with flow-based connectivity constraints.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: set of substations from `network_nodes.csv`.
    - Edges: set of candidate undirected links from `network_edges.csv`.
    - Directed Arcs: for each undirected edge, both (i, j) and (j, i) directions.
4.  **Define Decision Variables:**
    -   `y_e` = 1 if undirected link e is built (e ∈ Edges), 0 otherwise. Type: GRB.BINARY.
    -   `f_ij` = amount of auxiliary flow sent from root to node j via directed arc (i, j) (for all directed arcs). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Construction costs for each link: from `ConstructionCost` in `network_edges.csv`.
    -   Node list: from `Node` in `network_nodes.csv`.
    -   Edge list: from `Node1`, `Node2` in `network_edges.csv`.
    -   Root node: from `Value` where `Parameter` = 'RootNode' in `network_parameters.csv`.
6.  **Formulate Objective:** Minimize the total construction cost, i.e., sum over all candidate links of (`ConstructionCost` * `y_e`).
7.  **Formulate Constraints:**
    -   Constraint 1 (Spanning Tree Size): The total number of selected links equals n-1, where n is the number of nodes.
    -   Constraint 2 (Flow Conservation): For each node except the root, the net inflow of auxiliary flow equals 1 (i.e., each node receives one unit of flow from the root); for the root, the net outflow equals n-1.
    -   Constraint 3 (Flow-Linking): For each directed arc (i, j), the auxiliary flow `f_ij` is less than or equal to (n-1) times the binary variable `y_e` for the corresponding undirected edge e = {i, j}, ensuring flow can only traverse built links.
    -   Constraint 4 (Nonnegativity): All auxiliary flow variables `f_ij` are nonnegative.
    -   Constraint 5 (Binary): All link selection variables `y_e` are binary (0 or 1).
[Abstract Model Plan END]
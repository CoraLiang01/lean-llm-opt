[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a minimum-cost communication backbone (minimum spanning tree) that connects all substations (nodes) using a subset of available undirected links, ensuring network connectivity. The model must select exactly n-1 links, enforce connectivity via a flow-based formulation from a specified root node, and use binary and continuous variables as described.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree (MST) with flow-based connectivity constraints.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: All substations listed in `network_nodes.csv` (set N, size n = 8).
    - Edges: All candidate undirected links from `network_edges.csv` (set E, 14 edges).
    - For flow variables, each undirected edge is considered in both directions (ordered pairs of nodes for each edge).
4.  **Define Decision Variables:**
    -   `y_e` = 1 if undirected link e is built (where e = (i, j) from E), 0 otherwise. Type: GRB.BINARY.
    -   `f_ij` = amount of auxiliary flow sent from root node to node j via directed arc (i, j), for each direction of each candidate edge. Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Construction costs for each link: from `network_edges.csv`, column 'ConstructionCost'.
    -   Node list: from `network_nodes.csv`, column 'Node'.
    -   Edge list: from `network_edges.csv`, columns 'Node1', 'Node2'.
    -   Root node: from `network_parameters.csv`, row where 'Parameter' == 'RootNode', value in 'Value'.
6.  **Formulate Objective:** Minimize the total construction cost, i.e., sum over all candidate links e of (ConstructionCost_e * y_e).
7.  **Formulate Constraints:**
    -   Constraint 1 (Spanning Tree Cardinality): The number of selected links must be exactly n-1, i.e., sum over all e of y_e = n-1.
    -   Constraint 2 (Flow Conservation for Connectivity): For each node k ≠ root, the net inflow of auxiliary flow at node k must be 1 (i.e., sum of incoming flows minus outgoing flows = 1). For the root node, the net outflow must be n-1 (i.e., sum of outgoing flows minus incoming flows = n-1). For all other nodes, net flow is zero.
    -   Constraint 3 (Flow-Edge Linking): For each directed arc (i, j) corresponding to an undirected edge e, the auxiliary flow f_ij ≤ (n-1) * y_e. This ensures flow can only pass on selected links.
    -   Constraint 4 (Nonnegativity): All flow variables f_ij ≥ 0.
    -   Constraint 5 (Binary Restrictions): All link selection variables y_e ∈ {0, 1}.
[Abstract Model Plan END]
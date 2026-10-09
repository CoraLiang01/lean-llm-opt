[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a minimum-cost communication backbone that connects all substations (nodes) using a subset of available undirected links, ensuring the network is connected (spanning tree), with explicit flow-based connectivity constraints rooted at a specified node. The model must select exactly n-1 links, minimize total construction cost, and use auxiliary flow variables to enforce connectivity.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree (MST) formulation with flow-based connectivity constraints.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: set of substations, from `network_nodes.csv` (indexed by i, j).
    - Edges: set of candidate undirected links, from `network_edges.csv` (indexed by e, with endpoints Node1 and Node2).
    - Directed arcs: for each undirected edge, both (i, j) and (j, i) directions are considered for flow variables.
4.  **Define Decision Variables:**
    -   `y_e` = 1 if undirected link e is built (between Node1 and Node2), 0 otherwise. Type: GRB.BINARY.
    -   `f_ij` = amount of auxiliary flow sent from root node to node j along directed arc (i, j). Type: GRB.CONTINUOUS, lower bound 0.
5.  **Identify Parameters (from Schema):**
    -   Construction cost for each link: from `network_edges.csv`, column 'ConstructionCost'.
    -   Node set: from `network_nodes.csv`, column 'Node'.
    -   Edge set: from `network_edges.csv`, columns 'Node1', 'Node2'.
    -   Root node: from `network_parameters.csv`, row where 'Parameter' = 'RootNode', value in 'Value'.
    -   Number of nodes n: count of rows in `network_nodes.csv`.
6.  **Formulate Objective:** Minimize the total construction cost, i.e., sum over all candidate links of (ConstructionCost_e * y_e).
7.  **Formulate Constraints:**
    -   Constraint 1 (Spanning Tree Cardinality): The total number of selected links is exactly n-1, i.e., sum over all e of y_e = n-1.
    -   Constraint 2 (Flow Conservation for Connectivity): For each node j ≠ root, the net inflow of auxiliary flow is 1 (i.e., sum of incoming flows minus outgoing flows equals 1); for the root node, the net outflow is n-1 (i.e., sum of outgoing flows minus incoming flows equals n-1).
    -   Constraint 3 (Flow Capacity Linking): For each directed arc (i, j) corresponding to undirected edge e, the auxiliary flow f_ij ≤ (n-1) * y_e, ensuring flow can only pass on selected links.
    -   Constraint 4 (Nonnegativity): All flow variables f_ij ≥ 0.
    -   Constraint 5 (Binary Restrictions): All link selection variables y_e ∈ {0,1}.
[Abstract Model Plan END]
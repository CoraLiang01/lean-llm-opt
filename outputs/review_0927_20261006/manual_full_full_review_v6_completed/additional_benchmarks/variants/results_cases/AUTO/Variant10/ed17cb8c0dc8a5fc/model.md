[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a minimum-cost communication backbone that connects all substations (nodes) by selecting a subset of undirected links (edges) to form a spanning tree. The model must ensure connectivity using a flow-based formulation, select exactly n-1 links, and minimize total construction cost. Auxiliary directed flow variables are used to enforce connectivity from a designated root node.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree (MST) with flow-based connectivity constraints.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: set of substations, indexed by i, j (from network_nodes.csv).
    - Edges: set of candidate undirected links, indexed by e = (i, j) (from network_edges.csv).
    - Directed Arcs: for each undirected edge (i, j), both (i, j) and (j, i) are considered for flow variables.
4.  **Define Decision Variables:**
    - `y_e` = 1 if undirected link e is built, 0 otherwise. Type: GRB.BINARY.
    - `f_ij` = amount of auxiliary flow sent from root node to node j along directed arc (i, j). Type: GRB.CONTINUOUS, lower bound 0.
5.  **Identify Parameters (from Schema):**
    - Construction costs for each edge: 'ConstructionCost' column in network_edges.csv.
    - Node list: 'Node' column in network_nodes.csv (used to determine n and root node).
    - Edge-node mapping: 'Node1', 'Node2' columns in network_edges.csv.
    - Root node: 'Value' column in network_parameters.csv where 'Parameter' = 'RootNode'.
6.  **Formulate Objective:** Minimize the total construction cost, i.e., sum over all candidate edges of (ConstructionCost for edge e) × y_e.
7.  **Formulate Constraints:**
    - Constraint 1 (Spanning Tree Size): The total number of selected links is exactly n-1, where n is the number of nodes: sum over all edges of y_e = n-1.
    - Constraint 2 (Flow Conservation for Connectivity): For each node k ≠ root, the net inflow of auxiliary flow at node k is 1 (i.e., sum of incoming flows minus outgoing flows equals 1); for the root node, the net outflow is n-1 (i.e., sum of outgoing flows minus incoming flows equals n-1).
    - Constraint 3 (Flow-Edge Linking): For each directed arc (i, j), the auxiliary flow f_ij is less than or equal to (n-1) × y_e, where y_e is the binary variable for the undirected edge {i, j} (i.e., flow can only occur on selected edges, in either direction).
    - Constraint 4 (Nonnegativity): All flow variables f_ij ≥ 0.
    - Constraint 5 (Binary Restrictions): All link selection variables y_e ∈ {0, 1}.
[Abstract Model Plan END]
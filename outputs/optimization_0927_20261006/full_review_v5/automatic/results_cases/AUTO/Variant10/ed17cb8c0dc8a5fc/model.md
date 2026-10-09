[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a minimum-cost communication backbone connecting all substations (nodes) by selecting a subset of undirected links (edges) to form a minimum spanning tree. Connectivity is enforced using a flow-based formulation, with auxiliary directed flow variables from a specified root node to all others.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree (MST) with flow-based connectivity constraints.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: set of substations from `network_nodes.csv` (denote as N)
    - Edges: set of undirected candidate links from `network_edges.csv` (denote as E)
    - For each undirected edge, both directed arcs (i,j) and (j,i) are considered for flow variables.
4.  **Define Decision Variables:**
    - `y_e` = 1 if undirected link e ∈ E is built, 0 otherwise. Type: GRB.BINARY.
    - `f_ij` = amount of auxiliary flow sent from root node to node j along directed arc (i,j), for all arcs corresponding to candidate edges. Type: GRB.CONTINUOUS, lower bound 0.
5.  **Identify Parameters (from Schema):**
    - Construction costs for each edge: from `network_edges.csv` column 'ConstructionCost'.
    - Node list: from `network_nodes.csv` column 'Node'.
    - Edge-node incidence: from `network_edges.csv` columns 'Node1', 'Node2'.
    - Root node: from `network_parameters.csv` row where 'Parameter' = 'RootNode', value in 'Value'.
6.  **Formulate Objective:** Minimize total construction cost: sum over all edges e of ('ConstructionCost' for e) × y_e.
7.  **Formulate Constraints:**
    - Constraint 1 (Spanning Tree Cardinality): The number of selected links is exactly n-1, where n = |N|: sum over all edges e of y_e = n-1.
    - Constraint 2 (Flow Conservation for Connectivity): For each node k ≠ root, the net inflow of auxiliary flow at node k is 1 (i.e., sum of incoming f_ik minus sum of outgoing f_kj over all adjacent arcs equals 1); for the root node, the net outflow is n-1 (i.e., sum of outgoing f_root,j minus sum of incoming f_i,root equals n-1); for all other nodes, net flow is zero.
    - Constraint 3 (Flow-Edge Linking): For each directed arc (i,j) corresponding to undirected edge e, the auxiliary flow f_ij ≤ (n-1) × y_e, ensuring flow can only traverse selected links.
    - Constraint 4 (Nonnegativity): All flow variables f_ij ≥ 0.
    - Constraint 5 (Binary Restrictions): All link selection variables y_e ∈ {0,1}.
[Abstract Model Plan END]
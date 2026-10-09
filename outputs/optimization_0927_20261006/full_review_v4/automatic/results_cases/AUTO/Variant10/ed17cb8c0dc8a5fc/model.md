[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a minimum-cost communication backbone that connects all substations (nodes) using a subset of available undirected links, ensuring the network is connected (spanning tree), by selecting links to build and enforcing connectivity via a flow-based formulation rooted at a specified node.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree (MST) model with flow-based connectivity constraints.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: set of substations from `network_nodes.csv` (denote as N)
    - Edges: set of candidate undirected links from `network_edges.csv` (denote as E)
    - For each undirected edge, both directions are considered for flow variables (ordered pairs of nodes for each edge)
4.  **Define Decision Variables:**
    - `y_e` = 1 if undirected link e ∈ E is built, 0 otherwise. Type: GRB.BINARY.
    - `f_ij` = amount of auxiliary flow sent from root node to node j via directed arc (i, j), for each direction of each candidate edge. Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    - Construction costs for each link: from `network_edges.csv` column 'ConstructionCost' (indexed by edge e).
    - Node list: from `network_nodes.csv` column 'Node' (for set N).
    - Edge list: from `network_edges.csv` columns 'Node1', 'Node2' (for set E).
    - Root node: from `network_parameters.csv` row where 'Parameter' = 'RootNode', value in 'Value'.
6.  **Formulate Objective:** Minimize the total construction cost, i.e., minimize the sum over all candidate links of (ConstructionCost of e) × y_e.
7.  **Formulate Constraints:**
    - Constraint 1 (Spanning Tree Cardinality): The total number of selected links is exactly n-1, where n = |N| (sum over all e of y_e = n-1).
    - Constraint 2 (Flow Conservation for Connectivity): For each node k ≠ root, the net inflow of auxiliary flow at node k is exactly 1 (i.e., sum of incoming flows to k minus sum of outgoing flows from k equals 1); for the root node, the net outflow is n-1 (sum of outgoing flows from root minus sum of incoming flows to root equals n-1).
    - Constraint 3 (Flow-Linking): For each directed arc (i, j) corresponding to an undirected edge e, the auxiliary flow f_ij is bounded above by (n-1) × y_e (i.e., f_ij ≤ (n-1) × y_e), ensuring flow can only pass on selected links.
    - Constraint 4 (Nonnegativity): All flow variables f_ij ≥ 0.
    - Constraint 5 (Binary Restrictions): All link selection variables y_e ∈ {0,1}.
[Abstract Model Plan END]
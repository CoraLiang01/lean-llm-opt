[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum spanning tree (MST) network-design model for a utility company, selecting a least-cost set of links to connect all substations (nodes) using binary link-selection variables and flow-based connectivity constraints. The model must ensure exactly n-1 links are selected, all nodes are connected, and auxiliary flow variables are used to enforce connectivity from a specified root node.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree with flow-based connectivity enforcement.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: All substations listed in `network_nodes.csv` (set N).
    - Edges: All undirected candidate links in `network_edges.csv` (set E), each between two nodes.
    - Directed Arcs: For each undirected edge, both directions are considered for flow variables (set A).
4.  **Define Decision Variables:**
    -   `y_e` = 1 if undirected link e is built (between nodes i and j), 0 otherwise. Type: GRB.BINARY.
    -   `f_ij` = amount of auxiliary flow sent from root node to node j via directed arc (i, j). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Construction costs for each link: from `network_edges.csv`, column 'ConstructionCost'.
    -   Node list: from `network_nodes.csv`, column 'Node'.
    -   Edge list: from `network_edges.csv`, columns 'Node1', 'Node2'.
    -   Root node: from `network_parameters.csv`, row where 'Parameter' = 'RootNode', value in 'Value'.
6.  **Formulate Objective:** Minimize the total construction cost, i.e., sum over all candidate links of (ConstructionCost * y_e), where y_e indicates if link e is built.
7.  **Formulate Constraints:**
    -   Constraint 1 (Spanning Tree Cardinality): The total number of selected links must be exactly n-1, where n is the number of nodes: sum over all e of y_e = n-1.
    -   Constraint 2 (Flow Conservation for Connectivity): For each node k ≠ root, the net inflow of auxiliary flow must be 1 (i.e., each node receives one unit of flow from the root); for the root node, the net outflow must be n-1 (i.e., sends one unit to each other node). For all other nodes, net inflow minus outflow = 1; for root, outflow minus inflow = n-1.
    -   Constraint 3 (Flow-Linking): For each directed arc (i, j) corresponding to undirected edge e, the auxiliary flow f_ij ≤ (n-1) * y_e. This ensures flow can only pass on selected links.
    -   Constraint 4 (Nonnegativity): All flow variables f_ij ≥ 0.
    -   Constraint 5 (Binary Restrictions): All link-selection variables y_e ∈ {0,1}.
[Abstract Model Plan END]
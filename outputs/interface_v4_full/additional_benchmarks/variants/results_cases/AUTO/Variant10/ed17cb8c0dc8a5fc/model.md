[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a minimum-cost communication backbone (minimum spanning tree) that connects all substations (nodes) in a region. The model must select a subset of undirected links to build, ensuring all nodes are connected, using a flow-based connectivity formulation rooted at a specified node. The model must minimize total construction cost, select exactly n-1 links (where n is the number of nodes), and enforce connectivity via auxiliary flow variables.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree (MST) with flow-based connectivity constraints.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: set of substations, from `network_nodes.csv` (all 8 rows, column 'Node').
    - Edges: set of candidate undirected links, from `network_edges.csv` (all 14 rows, columns 'Node1', 'Node2').
    - For flow variables: ordered pairs (i, j) for each direction of each undirected edge.
4.  **Define Decision Variables:**
    -   `y_e` = 1 if undirected link e (between nodes i and j) is built, 0 otherwise. Type: GRB.BINARY.
    -   `f_ij` = amount of auxiliary flow sent from the root node to node j via directed arc (i, j) (for each direction of each candidate edge). Type: GRB.CONTINUOUS, lower bound 0.
5.  **Identify Parameters (from Schema):**
    -   Construction costs for each link: from `network_edges.csv`, column 'ConstructionCost'.
    -   Node list: from `network_nodes.csv`, column 'Node'.
    -   Edge list: from `network_edges.csv`, columns 'Node1', 'Node2'.
    -   Root node: from `network_parameters.csv`, row where 'Parameter' == 'RootNode', value in 'Value'.
6.  **Formulate Objective:** Minimize the total construction cost, i.e., sum over all candidate links of (ConstructionCost * y_e), where y_e indicates if link e is built.
7.  **Formulate Constraints:**
    -   **Link Count Constraint:** The total number of selected links must be exactly n-1, where n is the number of nodes (from `network_nodes.csv`): sum over all e of y_e = n-1.
    -   **Flow Conservation (Connectivity) Constraints:** For each node k ≠ root:
        - The net inflow of auxiliary flow to node k (sum of incoming f_ik minus sum of outgoing f_kj over all adjacent nodes) must be 1 (i.e., one unit of flow must reach each non-root node from the root).
        - For the root node: the net outflow must be n-1 (i.e., send one unit to each other node).
    -   **Flow-Linking Constraints:** For each directed arc (i, j) corresponding to an undirected edge e:
        - The auxiliary flow f_ij ≤ (n-1) * y_e, ensuring flow can only pass on built links.
    -   **Nonnegativity of Flow:** All f_ij ≥ 0.
    -   **Binary Restrictions:** All y_e ∈ {0, 1}.
[Abstract Model Plan END]
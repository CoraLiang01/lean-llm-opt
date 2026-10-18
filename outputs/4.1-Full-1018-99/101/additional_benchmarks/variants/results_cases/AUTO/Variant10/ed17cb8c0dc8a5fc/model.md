[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a minimum-cost communication backbone that connects all substations (nodes) using a subset of available undirected links, ensuring the network is connected (spanning tree), with exactly n-1 links selected, and connectivity enforced via a flow-based formulation from a specified root node.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree (MST) model with flow-based connectivity constraints.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: set of substations, from `network_nodes.csv` (field: 'Node').
    - Edges: set of candidate undirected links, from `network_edges.csv` (fields: 'Node1', 'Node2').
    - Directed Arcs: for each undirected edge, both directions (i,j) and (j,i).
4.  **Define Decision Variables:**
    - `y_e` = 1 if undirected link e is built (between nodes i and j), 0 otherwise. Type: GRB.BINARY.
    - `f_ij` = amount of auxiliary flow sent from root node to node j via directed arc (i,j). Type: GRB.CONTINUOUS, lower bound 0.
5.  **Identify Parameters (from Schema):**
    - Construction cost for each link: from `network_edges.csv` ('ConstructionCost').
    - Node set: from `network_nodes.csv` ('Node').
    - Edge list: from `network_edges.csv` ('Node1', 'Node2').
    - Root node: from `network_parameters.csv` ('Parameter' = 'RootNode', 'Value').
6.  **Formulate Objective:** Minimize the total construction cost, i.e., sum over all candidate links of (ConstructionCost * y_e).
7.  **Formulate Constraints:**
    - Constraint 1 (Spanning Tree Size): The total number of selected links equals n-1, where n is the number of nodes.
    - Constraint 2 (Flow Conservation for Connectivity): For each node other than the root, require that one unit of flow is received from the root. For the root node, the total flow sent out equals n-1.
    - Constraint 3 (Flow-Edge Linking): For each directed arc (i,j), the flow f_ij cannot exceed (n-1) * y_e, where y_e is the binary variable for the undirected link between i and j.
    - Constraint 4 (Nonnegativity): All flow variables f_ij are nonnegative.
    - Constraint 5 (Binary Restrictions): All link selection variables y_e are binary (0 or 1).
[Abstract Model Plan END]
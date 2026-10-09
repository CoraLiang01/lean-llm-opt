[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to model the minimum-cost construction of a connected backbone network (spanning tree) over all listed sites, using a single-commodity flow formulation. The model must select edges to connect all nodes, minimize total construction cost, and ensure connectivity via flow constraints from a designated root node.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree with single-commodity flow connectivity enforcement.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: set \( N \) (from all rows in network_nodes.csv, column 'Node')
    - Undirected edges: set \( E \) (from all rows in network_edges.csv, columns 'Node1', 'Node2')
    - Directed arcs: for each undirected edge \( (i, j) \in E \), both \( (i, j) \) and \( (j, i) \)
4.  **Define Decision Variables:**
    - \( y_{ij} \) = 1 if undirected edge between nodes \( i \) and \( j \) is selected, 0 otherwise. Type: GRB.BINARY.
    - \( f_{ij} \) = nonnegative flow sent from the root along directed arc \( (i, j) \). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    - Construction costs for each undirected edge: from network_edges.csv, column 'ConstructionCost'.
    - Node set: from network_nodes.csv, column 'Node'.
    - Root node: from network_parameters.csv, row where 'Parameter' = 'RootNode', value in 'Value'.
    - Number of nodes \( n \): count of rows in network_nodes.csv.
    - For flow-to-edge linking, \( M = n-1 \).
6.  **Formulate Objective:** Minimize the total construction cost, i.e., the sum over all undirected edges of (ConstructionCost × \( y_{ij} \)), where \( y_{ij} \) indicates if edge \( (i, j) \) is selected.
7.  **Formulate Constraints:**
    - Constraint 1 (Edge Count): The total number of selected edges is exactly \( n-1 \), i.e., sum over all \( y_{ij} \) = \( n-1 \).
    - Constraint 2 (Root Flow Balance): For the root node, the total flow sent out equals \( n-1 \).
    - Constraint 3 (Non-root Flow Balance): For each non-root node, the net flow in minus flow out equals 1 (each must receive one unit of flow from the root).
    - Constraint 4 (Flow-to-Edge Linking): For each directed arc \( (i, j) \), the flow \( f_{ij} \) is less than or equal to \( M \times y_{ij} \), ensuring flow only on selected edges.
    - Constraint 5 (Nonnegativity): All flow variables \( f_{ij} \) are nonnegative.
    - Constraint 6 (Binary Edge Selection): All \( y_{ij} \) variables are binary (0 or 1).
[Abstract Model Plan END]
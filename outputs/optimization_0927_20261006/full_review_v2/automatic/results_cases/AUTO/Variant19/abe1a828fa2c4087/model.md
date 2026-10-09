[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a set of undirected links (edges) to connect all listed sites (nodes) into a minimum-cost spanning tree, using a single-commodity flow formulation rooted at a specified node. The model must ensure connectivity, select exactly n-1 edges, and link flow variables to edge selection.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree with connectivity enforced via single-commodity flow.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: set \( N \) (from all rows in network_nodes.csv, column 'Node')
    - Undirected Edges: set \( E \) (from all rows in network_edges.csv, columns 'Node1', 'Node2')
    - Directed Arcs: for each undirected edge \( (i, j) \in E \), both \( (i, j) \) and \( (j, i) \)
4.  **Define Decision Variables:**
    - \( y_{ij} \) = 1 if undirected edge \( (i, j) \) is selected in the spanning tree, 0 otherwise. Type: GRB.BINARY.
    - \( f_{ij} \) = nonnegative flow sent from the root along directed arc \( (i, j) \). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    - Construction costs for each undirected edge: from network_edges.csv, column 'ConstructionCost'.
    - Node set: from network_nodes.csv, column 'Node'.
    - Root node: from network_parameters.csv, row where 'Parameter' = 'RootNode', value in 'Value'.
    - Number of nodes \( n \): count of rows in network_nodes.csv.
    - For flow-to-edge linking, \( M = n-1 \).
6.  **Formulate Objective:** Minimize the total construction cost of selected edges: sum over all undirected edges \( (i, j) \) of ConstructionCost\(_{ij}\) × \( y_{ij} \).
7.  **Formulate Constraints:**
    - Constraint 1 (Edge Count): The total number of selected edges is exactly \( n-1 \): sum over all \( y_{ij} \) = \( n-1 \).
    - Constraint 2 (Root Flow Balance): At the root node, the total flow sent out equals \( n-1 \): sum over all arcs leaving the root \( f_{root, j} \) = \( n-1 \).
    - Constraint 3 (Non-root Flow Balance): For each non-root node \( k \), the net flow in minus flow out equals 1: sum over all arcs into \( k \) minus sum over all arcs out of \( k \) = 1.
    - Constraint 4 (Flow-to-Edge Linking): For each directed arc \( (i, j) \), \( f_{ij} \leq (n-1) \cdot y_{ij} \), where \( y_{ij} \) is the binary variable for the undirected edge containing \( (i, j) \).
    - Constraint 5 (Variable Domains): \( y_{ij} \) are binary; \( f_{ij} \geq 0 \) for all directed arcs.
[Abstract Model Plan END]
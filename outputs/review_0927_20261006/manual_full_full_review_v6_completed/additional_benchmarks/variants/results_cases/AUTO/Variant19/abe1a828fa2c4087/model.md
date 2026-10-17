[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost spanning tree model for a network of sites, using a single-commodity flow formulation. The model must select a subset of undirected edges to connect all nodes (sites) at minimum total construction cost, ensuring the network is connected via flow from a designated root node. The model must use binary edge-selection variables, directed flow variables, and include constraints for edge count, flow conservation, and linking flow to edge selection.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree (MST) with single-commodity flow formulation.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: set of all sites from `network_nodes.csv` (denoted as N).
    - Undirected Edges: set of available links from `network_edges.csv` (denoted as E).
    - Directed Arcs: for each undirected edge (i, j) in E, both (i, j) and (j, i) are considered as directed arcs (denoted as A).
4.  **Define Decision Variables:**
    - `y_ij` = 1 if undirected edge (i, j) is selected in the spanning tree, 0 otherwise. Type: GRB.BINARY.
    - `f_ij` = amount of flow sent from the root along directed arc (i, j). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    - Construction costs for each undirected edge from `network_edges.csv` column 'ConstructionCost'.
    - Node set from `network_nodes.csv` column 'Node'.
    - Root node from `network_parameters.csv` where 'Parameter' = 'RootNode'.
    - Number of nodes n = total rows in `network_nodes.csv`.
    - For flow-to-edge linking, M = n - 1 (as specified in the query).
6.  **Formulate Objective:** Minimize the total construction cost of selected edges: sum over all undirected edges (i, j) of `ConstructionCost[i,j] * y_ij`.
7.  **Formulate Constraints:**
    - Constraint 1 (Edge Count): The number of selected edges must be exactly n - 1: sum over all (i, j) in E of y_ij = n - 1.
    - Constraint 2 (Root Node Flow Balance): At the root node r, the total flow sent out equals n - 1: sum over all arcs (r, j) of f_rj = n - 1.
    - Constraint 3 (Non-root Node Flow Balance): For each non-root node k, the net flow in minus flow out equals 1: sum over all arcs (i, k) of f_ik minus sum over all arcs (k, j) of f_kj = 1.
    - Constraint 4 (Flow-to-Edge Linking): For each directed arc (i, j), the flow f_ij cannot exceed M times the corresponding undirected edge selection variable: f_ij ≤ (n - 1) * y_ij, where y_ij corresponds to the undirected edge {i, j}.
    - Constraint 5 (Nonnegativity): All flow variables f_ij ≥ 0.
    - Constraint 6 (Binary Edge Selection): All y_ij ∈ {0, 1}.
[Abstract Model Plan END]
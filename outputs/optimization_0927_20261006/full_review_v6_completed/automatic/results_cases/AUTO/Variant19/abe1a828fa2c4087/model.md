[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost spanning tree model for a network of sites, ensuring all nodes are connected via selected undirected edges, using a single-commodity flow formulation rooted at a specified node. The model must minimize total construction cost, select exactly n-1 edges, enforce flow-based connectivity, and link flow to edge selection.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a minimum spanning tree (MST) with single-commodity flow connectivity constraints.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: set \( N \), from all rows in `network_nodes.csv` (column 'Node').
    - Undirected Edges: set \( E \), from all rows in `network_edges.csv` (columns 'Node1', 'Node2').
    - Directed Arcs: for each undirected edge, both directions (i,j) and (j,i).
4.  **Define Decision Variables:**
    - \( y_{ij} \) = 1 if undirected edge between nodes i and j is selected in the spanning tree; 0 otherwise. Type: GRB.BINARY.
    - \( f_{ij} \) = nonnegative flow sent from the root along directed arc (i, j). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    - Construction costs for each undirected edge from `network_edges.csv` ('ConstructionCost').
    - Node set from `network_nodes.csv` ('Node').
    - Root node from `network_parameters.csv` ('RootNode' in 'Value').
    - Number of nodes \( n \) is the count of rows in `network_nodes.csv`.
    - For flow-to-edge linking, \( M = n-1 \) (maximum possible flow on any arc).
6.  **Formulate Objective:** Minimize the total construction cost of selected edges: sum over all undirected edges of ('ConstructionCost' * \( y_{ij} \)).
7.  **Formulate Constraints:**
    - Constraint 1 (Edge Count): The total number of selected edges is exactly \( n-1 \): sum over all \( y_{ij} \) = \( n-1 \).
    - Constraint 2 (Root Flow Balance): At the root node, total flow sent out equals \( n-1 \): sum over all outgoing arcs from root of \( f_{root,j} \) = \( n-1 \).
    - Constraint 3 (Non-root Flow Balance): For each non-root node \( k \), net flow in minus flow out equals 1: sum over all incoming arcs to \( k \) of \( f_{i,k} \) minus sum over all outgoing arcs from \( k \) of \( f_{k,j} \) = 1.
    - Constraint 4 (Flow-to-Edge Linking): For each directed arc (i,j), \( f_{ij} \leq (n-1) \cdot y_{ij} \) (flow can only be sent on selected edges, and not more than \( n-1 \)).
    - Constraint 5 (Variable Domains): \( y_{ij} \) are binary; \( f_{ij} \geq 0 \).
[Abstract Model Plan END]
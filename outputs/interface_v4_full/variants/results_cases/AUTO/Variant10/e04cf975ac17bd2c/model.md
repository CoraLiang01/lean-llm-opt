[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a minimum-cost communication backbone (network) that connects all substations (nodes) using a subset of available undirected links, ensuring the network is connected (spanning tree), using a flow-based connectivity formulation rooted at a specified node. The model must select exactly n-1 links, enforce connectivity via flow variables, and minimize total construction cost.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) network design problem, specifically a Minimum Spanning Tree (MST) with flow-based connectivity constraints.
3.  **Define Index Sets:** The primary indices are:
    - Nodes: All substations listed in `network_nodes.csv` (set N, size n=8).
    - Edges: All candidate undirected links in `network_edges.csv` (set E, 14 edges).
    - For flow variables: Each direction of each undirected edge (i.e., for each edge {i,j}, both (i,j) and (j,i)).
4.  **Define Decision Variables:**
    -   `y_e` = 1 if undirected link e is built (where e = {i,j} ∈ E), 0 otherwise. Type: GRB.BINARY.
    -   `f_ij` = amount of auxiliary flow sent from root node to node j via directed arc (i,j), for all directed arcs corresponding to undirected edges. Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Construction costs for each link: from `network_edges.csv`, column 'ConstructionCost' for each edge {i,j}.
    -   Node set: from `network_nodes.csv`, column 'Node'.
    -   Root node: from `network_parameters.csv`, row where 'Parameter' = 'RootNode', value in 'Value'.
    -   Edge list: from `network_edges.csv`, columns 'Node1', 'Node2'.
6.  **Formulate Objective:** Minimize the total construction cost of selected links: sum over all edges e of `ConstructionCost[e] * y_e`.
7.  **Formulate Constraints:**
    -   **Spanning Tree Cardinality:** The number of selected links must be exactly n-1: sum over all edges e of y_e = n-1.
    -   **Flow Conservation (Connectivity):** For each node k ≠ root:
        - The net inflow of auxiliary flow to node k must be 1 (i.e., sum of incoming flows minus sum of outgoing flows = 1).
      For the root node:
        - The net outflow from the root must be n-1 (i.e., sum of outgoing flows minus sum of incoming flows = n-1).
      For all other nodes:
        - The net flow is -1 (i.e., sum of outgoing flows minus sum of incoming flows = -1).
    -   **Flow-Linking Constraints:** For each directed arc (i,j) corresponding to undirected edge {i,j}, the flow f_ij ≤ (n-1) * y_{i,j}. This ensures that flow can only be sent along selected links.
    -   **Nonnegativity:** All flow variables f_ij ≥ 0.
    -   **Binary Restrictions:** All link selection variables y_e ∈ {0,1}.
[Abstract Model Plan END]
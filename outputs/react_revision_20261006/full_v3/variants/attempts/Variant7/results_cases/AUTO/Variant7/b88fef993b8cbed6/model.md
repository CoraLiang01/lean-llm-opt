[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost transshipment model for a distribution network that ships goods from source plants to customers via intermediate cross-dock hubs. The model must account for source supply limits, customer demand requirements, hub throughput capacities, and per-unit transportation costs on each arc. The goal is to determine the optimal shipment flows on each arc to minimize total transportation cost.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) minimum-cost network flow (transshipment) problem.
3.  **Define Index Sets:** The primary indices are:
    - Sources (S): Nodes with 'NodeType' = 'SourceSupply' (from node_supply_demand.csv)
    - Customers (C): Nodes with 'NodeType' = 'CustomerDemand' (from node_supply_demand.csv)
    - Hubs (H): Nodes listed in hub_capacity.csv
    - Arcs (A): Directed arcs (i, j) with costs, as listed in arc_costs.csv (from 'From' to 'To')
4.  **Define Decision Variables:**
    -   `f_ij` = Amount of goods shipped on arc from node i to node j (for each arc (i, j) in arc_costs.csv). Type: GRB.CONTINUOUS (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit on each arc: from arc_costs.csv, column 'Cost' for each ('From', 'To') pair.
    -   Source supply limits: from node_supply_demand.csv, 'Amount' where 'NodeType' = 'SourceSupply'.
    -   Customer demand requirements: from node_supply_demand.csv, 'Amount' where 'NodeType' = 'CustomerDemand'.
    -   Hub throughput capacities: from hub_capacity.csv, 'ThroughputCapacity' for each hub.
6.  **Formulate Objective:** Minimize total transportation cost, i.e., sum over all arcs (i, j) of (arc_costs['Cost'] * f_ij).
7.  **Formulate Constraints:**
    -   Source supply upper-bound constraints: For each source s, the total flow out of s (sum over all arcs from s to any hub) ≤ source supply amount (from node_supply_demand.csv).
    -   Customer demand constraints: For each customer c, the total flow into c (sum over all arcs from any hub to c) ≥ customer demand amount (from node_supply_demand.csv).
    -   Hub flow-balance constraints: For each hub h, total flow into h (from all sources) = total flow out of h (to all customers).
    -   Hub throughput-capacity constraints: For each hub h, the total flow passing through h (either total in or total out, since they are equal) ≤ hub throughput capacity (from hub_capacity.csv).
    -   Nonnegativity constraints: For all arcs (i, j), f_ij ≥ 0.
[Abstract Model Plan END]
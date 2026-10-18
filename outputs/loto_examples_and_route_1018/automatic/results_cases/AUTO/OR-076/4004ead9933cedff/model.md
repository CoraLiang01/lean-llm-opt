[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select an optimal subset of potential warehouse locations to serve all customers, minimizing the total cost, which includes both the fixed annual opening costs of selected warehouses and the variable transportation costs for fulfilling all customer demand. Each warehouse has a maximum service capacity, and all customer demands must be fully satisfied.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location (fixed-charge) problem.
3.  **Define Index Sets:** The primary indices are:
    - Warehouses (indexed by $i$; from 'Warehouse ID' in warehouse.csv and cost.csv)
    - Customers (indexed by $j$; from 'Customer ID' in demand.csv and cost.csv)
4.  **Define Decision Variables:**
    -   $y_i$ = 1 if warehouse $i$ is opened, 0 otherwise. Type: GRB.BINARY.
    -   $x_{ij}$ = amount of customer $j$'s demand served from warehouse $i$. Type: GRB.CONTINUOUS (non-negative).
5.  **Identify Parameters (from Schema):**
    -   Fixed opening cost for warehouse $i$: from 'Fixed_Cost' in warehouse.csv.
    -   Warehouse capacity for $i$: from 'Capacity' in warehouse.csv.
    -   Customer demand for $j$: from 'Demand' in demand.csv.
    -   Transportation cost per unit from warehouse $i$ to customer $j$: from cost.csv, columns 'C1'...'C20' (mapping 'Warehouse ID' to $i$, 'Cj' to $j$).
6.  **Formulate Objective:** Minimize the sum of fixed opening costs for selected warehouses plus the total transportation cost for serving all customer demand, i.e., minimize $\sum_i \text{Fixed\_Cost}_i \cdot y_i + \sum_{i,j} \text{Cost}_{ij} \cdot x_{ij}$.
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each customer $j$, the sum of $x_{ij}$ over all warehouses $i$ must equal the total demand of customer $j$ (i.e., $\sum_i x_{ij} = \text{Demand}_j$).
    -   Warehouse Capacity: For each warehouse $i$, the total amount shipped from warehouse $i$ to all customers cannot exceed its capacity if it is opened (i.e., $\sum_j x_{ij} \leq \text{Capacity}_i \cdot y_i$).
    -   Non-negativity: $x_{ij} \geq 0$ for all $i, j$.
    -   Binary: $y_i \in \{0,1\}$ for all $i$.
[Abstract Model Plan END]
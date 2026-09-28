[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select an optimal subset of potential warehouse locations to open, and assign all customer demands to these warehouses, so that the total cost (fixed warehouse opening costs plus variable transportation costs) is minimized. Each warehouse has a maximum capacity, and all customer demands must be fully satisfied.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location (fixed-charge) problem.
3.  **Define Index Sets:** The primary indices are:
    - Warehouses (indexed by $i$), from 'Warehouse ID' in warehouse.csv and cost.csv.
    - Customers (indexed by $j$), from 'Customer ID' in demand.csv and cost.csv.
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if warehouse $i$ is opened, 0 otherwise. Type: GRB.BINARY.
    -   `x[i,j]` = amount of customer $j$'s demand served by warehouse $i$. Type: GRB.CONTINUOUS (non-negative).
5.  **Identify Parameters (from Schema):**
    -   Fixed opening cost for each warehouse: from 'Fixed_Cost' column in warehouse.csv.
    -   Warehouse capacity: from 'Capacity' column in warehouse.csv.
    -   Customer demand: from 'Demand' column in demand.csv.
    -   Transportation cost per unit from warehouse $i$ to customer $j$: from cost.csv, columns 'C1'...'C20' (each column corresponds to a customer, each row to a warehouse).
6.  **Formulate Objective:** Minimize the total cost, which is the sum of fixed opening costs for selected warehouses plus the sum of transportation costs for all assignments:
    - $\text{Minimize} \quad \sum_{i} \text{Fixed\_Cost}[i] \cdot y[i] + \sum_{i} \sum_{j} \text{Cost}[i,j] \cdot x[i,j]$
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer $j$, the sum of assignments from all warehouses must meet the customer's demand exactly:
        - $\sum_{i} x[i,j] = \text{Demand}[j]$ for all $j$.
    -   Constraint 2 (Warehouse Capacity): For each warehouse $i$, the total amount assigned from that warehouse to all customers cannot exceed its capacity, and only if the warehouse is open:
        - $\sum_{j} x[i,j] \leq \text{Capacity}[i] \cdot y[i]$ for all $i$.
    -   Constraint 3 (Assignment Only from Open Warehouses): No customer can be assigned to a warehouse unless it is open (enforced by capacity constraint above).
    -   Constraint 4 (Non-negativity): $x[i,j] \geq 0$ for all $i, j$.
    -   Constraint 5 (Binary): $y[i] \in \{0,1\}$ for all $i$.
[Abstract Model Plan END]
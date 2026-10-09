[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select an optimal subset of potential warehouse locations to serve all customers, minimizing the total cost, which includes both the fixed annual opening costs of warehouses and the variable transportation costs for fulfilling all customer demand, subject to warehouse capacity limits.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location (fixed-charge) model.
3.  **Define Index Sets:** The primary indices are:
    - Warehouses (indexed by $i$; from 'Warehouse ID' in warehouse.csv and cost.csv)
    - Customers (indexed by $j$; from 'Customer ID' in demand.csv and cost.csv)
4.  **Define Decision Variables:**
    - $y_i$ = 1 if warehouse $i$ is opened, 0 otherwise. Type: GRB.BINARY.
    - $x_{ij}$ = amount of customer $j$'s demand served from warehouse $i$. Type: GRB.CONTINUOUS (non-negative).
5.  **Identify Parameters (from Schema):**
    - Fixed opening cost for each warehouse: 'Fixed_Cost' from warehouse.csv.
    - Warehouse capacity: 'Capacity' from warehouse.csv.
    - Customer demand: 'Demand' from demand.csv.
    - Transportation cost per unit from warehouse $i$ to customer $j$: cost.csv, columns 'C1'–'C20' (each column corresponds to a customer, each row to a warehouse).
6.  **Formulate Objective:** Minimize the total cost, which is the sum of fixed opening costs for selected warehouses plus the sum of transportation costs for all customer assignments:
       - $\sum_{i} \text{Fixed\_Cost}_i \cdot y_i + \sum_{i} \sum_{j} \text{Cost}_{ij} \cdot x_{ij}$
7.  **Formulate Constraints:**
    - Demand Satisfaction: For each customer $j$, the total amount assigned from all warehouses must meet their demand exactly:
        - $\sum_{i} x_{ij} = \text{Demand}_j$ for all $j$.
    - Warehouse Capacity: For each warehouse $i$, the total amount shipped from that warehouse cannot exceed its capacity, and only if it is opened:
        - $\sum_{j} x_{ij} \leq \text{Capacity}_i \cdot y_i$ for all $i$.
    - Non-negativity: $x_{ij} \geq 0$ for all $i, j$.
    - Binary Opening: $y_i \in \{0,1\}$ for all $i$.
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select an optimal subset of potential warehouse locations to open, and assign all customer demand to these warehouses, so as to minimize the total cost (sum of fixed warehouse opening costs and variable transportation costs), subject to warehouse capacity limits and full demand satisfaction.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location (warehouse location) problem with fixed-charge and assignment structure.
3.  **Define Index Sets:** The primary indices are:
    - Warehouses (indexed by \( i \)), from all rows in warehouse.csv and cost.csv (Warehouse ID).
    - Customers (indexed by \( j \)), from all rows in demand.csv and cost.csv (Customer ID).
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if warehouse \( i \) is opened, 0 otherwise. Type: GRB.BINARY.
    -   `x[i,j]` = amount of customer \( j \)'s demand served from warehouse \( i \). Type: GRB.CONTINUOUS (non-negative).
5.  **Identify Parameters (from Schema):**
    -   Fixed opening cost for each warehouse: from warehouse.csv, column 'Fixed_Cost', keyed by 'Warehouse ID'.
    -   Warehouse capacity: from warehouse.csv, column 'Capacity', keyed by 'Warehouse ID'.
    -   Customer demand: from demand.csv, column 'Demand', keyed by 'Customer ID'.
    -   Transportation cost per unit from warehouse \( i \) to customer \( j \): from cost.csv, columns 'C1'...'C20' (customer columns), keyed by 'Warehouse ID' and 'Customer ID'.
6.  **Formulate Objective:** Minimize the total cost, which is the sum of:
    - Fixed opening costs for all selected warehouses: \(\sum_{i} \text{Fixed_Cost}[i] \cdot y[i]\)
    - Plus total transportation costs for all assignments: \(\sum_{i}\sum_{j} \text{Cost}[i,j] \cdot x[i,j]\)
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer \( j \), the sum of all assignments from all warehouses must meet their demand exactly: \(\sum_{i} x[i,j] = \text{Demand}[j]\).
    -   Constraint 2 (Warehouse Capacity): For each warehouse \( i \), the total amount assigned from that warehouse to all customers cannot exceed its capacity: \(\sum_{j} x[i,j] \leq \text{Capacity}[i]\).
    -   Constraint 3 (Assignment Only from Opened Warehouses): For each warehouse \( i \) and customer \( j \), assignments can only be made if the warehouse is open: \(x[i,j] \leq \text{Demand}[j] \cdot y[i]\) (or, more generally, \(x[i,j] \leq \text{Capacity}[i] \cdot y[i]\), but the demand-based bound is tighter and sufficient).
    -   Constraint 4 (Variable Domains): \(y[i] \in \{0,1\}\), \(x[i,j] \geq 0\).
[Abstract Model Plan END]
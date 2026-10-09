[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which suppliers to activate and how much each supplier should ship to each store, so that all store demands for liquor products are met at minimum total cost. The total cost includes both fixed supplier activation costs and per-unit transportation costs.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a Fixed-Charge Facility Location (Uncapacitated Facility Location) problem.
3.  **Define Index Sets:** The primary indices are:
    - Suppliers (from `fixed_cost.csv` and `transportation_costs.csv` rows)
    - Stores/Customers (from `demand.csv` and `transportation_costs.csv` columns)
4.  **Define Decision Variables:**
    - `x[i,j]` = Quantity of goods shipped from supplier `i` to store `j`. Type: GRB.CONTINUOUS (nonnegative real numbers).
    - `y[i]` = 1 if supplier `i` is activated (operational), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Fixed activation costs for each supplier: from `fixed_cost.csv` column `'fixed_costs'`, keyed by supplier name (`'Unnamed: 0'`).
    - Per-unit transportation costs: from `transportation_costs.csv`, with supplier as row (`'Unnamed: 0'`) and store as column.
    - Store demands: from `demand.csv` column `'demand'`, keyed by store (`'Customer'`).
6.  **Formulate Objective:** Minimize the total cost, which is the sum of:
    - The fixed activation cost for each supplier that is activated (`sum over i of fixed_cost[i] * y[i]`)
    - The total transportation cost for all shipments (`sum over i,j of transportation_cost[i,j] * x[i,j]`)
7.  **Formulate Constraints:**
    - Demand satisfaction: For each store `j`, the sum of shipments received from all suppliers must equal the store's demand (`sum over i of x[i,j] = demand[j]`).
    - Supplier activation linking: For each supplier `i` and store `j`, shipments from supplier `i` to store `j` are only allowed if supplier `i` is activated (`x[i,j] <= M * y[i]`, where `M` is a sufficiently large constant, e.g., the total demand).
    - Nonnegativity: All shipment variables must be nonnegative (`x[i,j] >= 0`).
    - Binary activation: Each supplier activation variable must be binary (`y[i]` in {0,1}).
[Abstract Model Plan END]
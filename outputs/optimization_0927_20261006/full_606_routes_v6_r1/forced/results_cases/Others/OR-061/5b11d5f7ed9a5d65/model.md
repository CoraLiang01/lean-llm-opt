[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine which suppliers to activate and how much each branch should source from each supplier, so that all branch demands are met and the total cost (fixed supplier activation costs plus transportation costs) is minimized.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a Fixed-Charge Facility Location (Uncapacitated Facility Location) problem.
3.  **Define Index Sets:** The primary indices are:
    - Suppliers (i ∈ Suppliers, from 'fixed_cost.csv' and 'transportation_costs.csv' rows)
    - Branches/Customers (j ∈ Branches, from 'demand.csv' and 'transportation_costs.csv' columns)
4.  **Define Decision Variables:**
    -   `x[i,j]` = Quantity of goods supplied from supplier i to branch j. Type: GRB.CONTINUOUS (≥ 0).
    -   `y[i]` = 1 if supplier i is activated (operational), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Fixed activation costs for each supplier: 'fixed_costs' column in 'fixed_cost.csv', keyed by supplier.
    -   Transportation cost per unit from each supplier to each branch: 'transportation_costs.csv', with suppliers as rows and branches as columns.
    -   Demand for each branch: 'demand' column in 'demand.csv', keyed by branch.
6.  **Formulate Objective:** Minimize the total cost, which is the sum of all supplier fixed activation costs (for activated suppliers) plus the sum of all transportation costs (quantity shipped times per-unit cost for each supplier-branch pair).
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each branch j, the sum over all suppliers i of x[i,j] must equal the demand of branch j (from 'demand.csv').
    -   Supplier Activation Linking: For each supplier i and branch j, x[i,j] ≤ M * y[i], where M is a sufficiently large constant (e.g., the total demand across all branches), ensuring that no goods are shipped from a supplier unless it is activated.
    -   Non-negativity: All x[i,j] ≥ 0.
    -   Binary Activation: All y[i] ∈ {0,1}.
[Abstract Model Plan END]
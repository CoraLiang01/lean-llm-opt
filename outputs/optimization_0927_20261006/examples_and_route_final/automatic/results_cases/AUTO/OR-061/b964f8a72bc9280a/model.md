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
    -   Fixed activation costs: from 'fixed_cost.csv', column 'fixed_costs', indexed by supplier.
    -   Transportation costs per unit: from 'transportation_costs.csv', columns 'C1'...'C5', indexed by supplier and branch.
    -   Branch demands: from 'demand.csv', column 'demand', indexed by branch.
6.  **Formulate Objective:** Minimize the total cost, which is the sum of all supplier fixed activation costs (for activated suppliers) plus the sum of all transportation costs (units shipped times per-unit cost) across all supplier-branch pairs.
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each branch j, the sum over all suppliers i of x[i,j] must equal the demand of branch j (from 'demand.csv').
    -   Supplier Activation Linking: For each supplier i and branch j, x[i,j] ≤ M[j] * y[i], where M[j] is a sufficiently large constant (e.g., the total demand of branch j), ensuring that no goods are shipped from a supplier unless it is activated.
    -   Non-negativity: All x[i,j] ≥ 0.
    -   Binary Activation: All y[i] ∈ {0,1}.
[Abstract Model Plan END]
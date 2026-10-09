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
    -   Fixed supplier activation costs: from 'fixed_cost.csv', column 'fixed_costs', keyed by supplier.
    -   Transportation costs per unit: from 'transportation_costs.csv', columns 'C1'...'C5' (branches), keyed by supplier (row 'Unnamed: 0').
    -   Branch demands: from 'demand.csv', column 'demand', keyed by branch (column 'customer').
6.  **Formulate Objective:** Minimize the total cost, which is the sum of all activated suppliers' fixed costs plus the sum of all transportation costs for goods shipped from suppliers to branches:
        - Total Cost = sum over i (fixed_costs[i] * y[i]) + sum over i,j (transportation_costs[i,j] * x[i,j])
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each branch j, the total goods received from all suppliers must equal its demand: sum over i (x[i,j]) = demand[j].
    -   Supplier Activation Linking: For each supplier i and branch j, only allow shipments from supplier i if it is activated: x[i,j] ≤ M * y[i], where M is a sufficiently large constant (e.g., sum of all demands).
    -   Non-negativity: x[i,j] ≥ 0 for all i, j.
    -   Binary Activation: y[i] ∈ {0,1} for all i.
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost capacitated lot-sizing model for a manufacturer planning monthly production of a spare part over 12 months, using data on demand, costs, and capacities from monthly_lot_sizing.csv. The model must ensure all demand is met on time (no backlogging), initial and final inventories are zero, and production is subject to monthly capacity and setup costs.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated lot-sizing problem with setup costs.
3.  **Define Index Sets:** The primary index is Months (t = 1, ..., 12), corresponding to the 12 rows in the CSV file.
4.  **Define Decision Variables:**
    -   `x[t]` = Production quantity in month t. Type: GRB.CONTINUOUS (nonnegative).
    -   `inv[t]` = Ending inventory at the end of month t. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[t]` = 1 if production is set up in month t, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   Production cost per unit: 'ProductionCost' column.
        -   Setup cost per month: 'SetupCost' column.
        -   Inventory holding cost per unit: 'HoldingCost' column.
    -   Constraint coefficients:
        -   Demand per month: 'Demand' column.
        -   Production capacity per month: 'ProductionCapacity' column.
    -   Constraint RHS:
        -   Demand per month: 'Demand' column.
        -   Production capacity per month: 'ProductionCapacity' column.
        -   Initial inventory: 0 (given).
        -   Final inventory: 0 (required).
6.  **Formulate Objective:** Minimize total cost over 12 months, which is the sum over all months of:
        -   (ProductionCost[t] * x[t]) + (SetupCost[t] * y[t]) + (HoldingCost[t] * inv[t])
7.  **Formulate Constraints:**
    -   Constraint 1 (Inventory Balance for each month t): 
        -   For t = 1: x[1] - Demand[1] = inv[1] (since initial inventory is 0)
        -   For t = 2 to 12: inv[t-1] + x[t] - Demand[t] = inv[t]
    -   Constraint 2 (Production Capacity and Setup Linking for each month t): 
        -   x[t] ≤ ProductionCapacity[t] * y[t] (production only if setup occurs, and cannot exceed capacity)
    -   Constraint 3 (No Backlogging): 
        -   inv[t] ≥ 0 for all t (inventory cannot be negative)
    -   Constraint 4 (Initial Inventory): 
        -   inv[0] = 0 (implicitly handled in the first inventory balance)
    -   Constraint 5 (Final Inventory): 
        -   inv[12] = 0 (ending inventory after last month must be zero)
    -   Constraint 6 (Nonnegativity and Binary Restrictions): 
        -   x[t] ≥ 0, inv[t] ≥ 0 for all t; y[t] ∈ {0,1} for all t
[Abstract Model Plan END]
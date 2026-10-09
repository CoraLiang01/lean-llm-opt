[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost capacitated lot-sizing model for a manufacturer planning monthly production of a spare part over 12 months, using data on demand, costs, and capacities from monthly_lot_sizing.csv. The model must ensure all demand is met on time (no backlogging), initial and final inventories are zero, and production is subject to monthly capacity and setup costs.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated lot-sizing problem with setup costs.
3.  **Define Index Sets:** The primary index is Months (t = 1 to 12), corresponding to the 12 rows in the CSV file.
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
6.  **Formulate Objective:** Minimize the total cost over all months, which is the sum of (production cost per unit * production quantity) + (setup cost if production occurs) + (holding cost per unit * ending inventory) for each month.
7.  **Formulate Constraints:**
    -   Constraint 1 (Inventory Balance): For each month t, ensure that starting inventory plus production minus demand equals ending inventory. For month 1, starting inventory is zero. For months 2–12, starting inventory is previous month's ending inventory.
    -   Constraint 2 (Production Capacity and Setup Linking): For each month t, production quantity x[t] cannot exceed production capacity in that month times y[t] (i.e., x[t] ≤ ProductionCapacity[t] * y[t]). This ensures setup cost is incurred if any production occurs.
    -   Constraint 3 (No Backlogging): Ending inventory inv[t] must be nonnegative for all months (no negative inventory).
    -   Constraint 4 (Initial Inventory): Inventory at the end of month 0 (before month 1) is zero.
    -   Constraint 5 (Final Inventory): Inventory at the end of month 12 (after last month) is zero.
    -   Constraint 6 (Variable Domains): x[t] ≥ 0 (continuous), inv[t] ≥ 0 (continuous), y[t] ∈ {0,1} (binary) for all months t.
[Abstract Model Plan END]
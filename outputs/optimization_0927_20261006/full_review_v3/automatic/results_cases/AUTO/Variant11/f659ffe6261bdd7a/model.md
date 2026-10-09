[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost capacitated lot-sizing model for monthly production planning of a spare part over 12 months, using monthly demand, production cost, setup cost, holding cost, and production capacity. The model must ensure no backlogging, zero initial and ending inventory, and link production to setup decisions.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated lot-sizing model with setup costs.
3.  **Define Index Sets:** The primary index is Months (t = 1, ..., 12), corresponding to the 'Month' column in the CSV.
4.  **Define Decision Variables:**
    -   `x[t]` = Production quantity in month t. Type: GRB.CONTINUOUS (nonnegative).
    -   `inv[t]` = Ending inventory at the end of month t. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[t]` = 1 if production is set up in month t, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   Production cost per unit: 'ProductionCost'
        -   Setup cost per month: 'SetupCost'
        -   Holding cost per unit: 'HoldingCost'
    -   Constraint coefficients:
        -   Demand per month: 'Demand'
        -   Production capacity per month: 'ProductionCapacity'
    -   Constraint RHS:
        -   Demand per month: 'Demand'
        -   Production capacity per month: 'ProductionCapacity'
6.  **Formulate Objective:** Minimize total cost over all months, where total cost is the sum of (production cost per unit * production quantity) + (setup cost if production occurs) + (holding cost per unit * ending inventory) for each month.
7.  **Formulate Constraints:**
    -   Constraint 1 (Inventory Balance): For each month t, ensure that starting inventory plus production minus demand equals ending inventory. For t=1, starting inventory is zero; for t>1, starting inventory is inv[t-1].
    -   Constraint 2 (Production Capacity and Setup Linking): For each month t, production quantity x[t] cannot exceed production capacity times y[t] (i.e., x[t] ≤ ProductionCapacity[t] * y[t]).
    -   Constraint 3 (No Backlogging): For each month t, ending inventory inv[t] ≥ 0.
    -   Constraint 4 (Nonnegativity): For each month t, production quantity x[t] ≥ 0.
    -   Constraint 5 (Binary Setup): For each month t, y[t] ∈ {0,1}.
    -   Constraint 6 (Zero Initial Inventory): inv[0] = 0 (implicitly, starting inventory before month 1 is zero).
    -   Constraint 7 (Zero Ending Inventory): inv[12] = 0 (ending inventory after the last month must be zero).
[Abstract Model Plan END]
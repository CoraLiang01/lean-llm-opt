[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost capacitated lot-sizing model for a manufacturer planning monthly production of a seasonal spare part over 12 months, using data on demand, production cost, setup cost, holding cost, and production capacity. The model must ensure no initial or final inventory, no backlogging, and must include inventory balance, capacity-to-setup linking, and appropriate variable restrictions.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated lot-sizing problem with setup costs.
3.  **Define Index Sets:** The primary index is Months (t = 1, ..., 12), corresponding to the 12 rows in the CSV file.
4.  **Define Decision Variables:**
    -   `x_t` = Production quantity in month t. Type: GRB.CONTINUOUS (nonnegative).
    -   `inv_t` = Ending inventory at the end of month t. Type: GRB.CONTINUOUS (nonnegative).
    -   `y_t` = 1 if production is set up in month t, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   Production cost per unit: 'ProductionCost' column.
        -   Setup cost per month: 'SetupCost' column.
        -   Holding cost per unit carried to next month: 'HoldingCost' column.
    -   Constraint coefficients:
        -   Demand per month: 'Demand' column.
        -   Production capacity per month: 'ProductionCapacity' column.
    -   Constraint RHS:
        -   Demand per month: 'Demand' column.
        -   Production capacity per month: 'ProductionCapacity' column.
        -   Initial inventory: 0 (given).
        -   Final inventory: 0 (required).
6.  **Formulate Objective:** Minimize total cost over all months, which is the sum of (production cost per unit * production quantity) + (setup cost if production occurs) + (holding cost per unit of ending inventory) for each month.
7.  **Formulate Constraints:**
    -   Constraint 1 (Inventory Balance): For each month t, ensure that starting inventory plus production minus demand equals ending inventory. For t=1, starting inventory is 0. For t>1, starting inventory is inv_{t-1}.
        - inv_{t-1} + x_t - Demand_t = inv_t, for t = 1,...,12 (with inv_0 = 0).
    -   Constraint 2 (Production Capacity and Setup Linking): Production in any month cannot exceed the month's capacity if and only if setup occurs.
        - x_t ≤ ProductionCapacity_t * y_t, for all t.
    -   Constraint 3 (No Backlogging): All inventory variables inv_t ≥ 0, and production variables x_t ≥ 0.
    -   Constraint 4 (No Initial Inventory): inv_0 = 0 (enforced in the first inventory balance).
    -   Constraint 5 (No Final Inventory): inv_{12} = 0.
    -   Constraint 6 (Binary Setup): y_t ∈ {0,1} for all t.
[Abstract Model Plan END]
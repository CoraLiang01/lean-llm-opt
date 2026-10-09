[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost capacitated lot-sizing model for monthly production planning of a spare part over 12 months, using data on demand, costs, and capacities from the CSV. The model must determine monthly production quantities, setups, and inventories, ensuring all demand is met on time, no backlogging, zero initial and final inventory, and capacity/setup constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated lot-sizing model with setup costs.
3.  **Define Index Sets:** The primary index is Months (t = 1, ..., 12), corresponding to the 'Month' column in the CSV.
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
        -   Demand fulfillment: 'Demand' column.
        -   Production upper bound: 'ProductionCapacity' column.
        -   Initial and final inventory: zero.
6.  **Formulate Objective:** Minimize total cost over all months, which is the sum of (production cost per unit * production quantity) + (setup cost if production occurs) + (holding cost per unit * ending inventory) for each month.
7.  **Formulate Constraints:**
    -   Constraint 1 (Inventory Balance): For each month t, ensure that starting inventory plus production minus demand equals ending inventory. For t=1, starting inventory is zero; for t>1, starting inventory is previous month's ending inventory.
    -   Constraint 2 (Production Capacity and Setup Linking): For each month t, production quantity cannot exceed production capacity times the setup binary variable (i.e., x[t] ≤ ProductionCapacity[t] * y[t]).
    -   Constraint 3 (No Backlogging): All variables x[t] and inv[t] are nonnegative; inv[t] ≥ 0 for all t.
    -   Constraint 4 (Zero Initial Inventory): inv[0] = 0 (implicitly, starting inventory before month 1 is zero).
    -   Constraint 5 (Zero Ending Inventory): inv[12] = 0 (ending inventory after the last month is zero).
    -   Constraint 6 (Binary Setup): y[t] ∈ {0,1} for all t.
[Abstract Model Plan END]
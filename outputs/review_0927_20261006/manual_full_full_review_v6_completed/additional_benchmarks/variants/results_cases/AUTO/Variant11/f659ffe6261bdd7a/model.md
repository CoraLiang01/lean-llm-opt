[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to formulate a minimum-cost capacitated lot-sizing model for monthly production planning of a spare part over 12 months, using monthly demand, production cost, setup cost, holding cost, and production capacity. The model must ensure no backlogging, zero initial and ending inventory, and link production to setup decisions.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated lot-sizing model with setup costs.
3.  **Define Index Sets:** The primary index is Months (t = 1, ..., 12), corresponding to the 'Month' column in the CSV.
4.  **Define Decision Variables:**
    -   `x[t]` = Production quantity in month t. Type: GRB.CONTINUOUS (nonnegative).
    -   `inv[t]` = Ending inventory at the end of month t. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[t]` = 1 if production is set up in month t, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Production cost per unit: 'ProductionCost' column.
    -   Setup cost per month: 'SetupCost' column.
    -   Holding cost per unit carried to next month: 'HoldingCost' column.
    -   Monthly demand: 'Demand' column.
    -   Monthly production capacity: 'ProductionCapacity' column.
6.  **Formulate Objective:** Minimize total cost over all months, which is the sum of (production cost per unit * production quantity) + (setup cost if production occurs) + (holding cost per unit of ending inventory) for each month.
7.  **Formulate Constraints:**
    -   Inventory Balance: For each month t, ensure that starting inventory plus production minus demand equals ending inventory. For t=1, starting inventory is zero; for t>1, starting inventory is inv[t-1].
    -   Production-Capacity-to-Setup Linking: For each month t, production quantity x[t] cannot exceed production capacity times y[t] (i.e., x[t] ≤ ProductionCapacity[t] * y[t]).
    -   No Backlogging: inv[t] ≥ 0 for all t.
    -   Initial Inventory: inv[0] = 0 (implicitly, starting inventory before month 1 is zero).
    -   Ending Inventory: inv[12] = 0 (ending inventory after the last month must be zero).
    -   Nonnegativity: x[t] ≥ 0, inv[t] ≥ 0 for all t.
    -   Binary Setup: y[t] ∈ {0,1} for all t.
[Abstract Model Plan END]
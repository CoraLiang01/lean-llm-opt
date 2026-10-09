[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine a 24-month production and inventory plan for a seasonal appliance that minimizes total costs (production, setup, and inventory holding), meeting monthly demand exactly, never allowing unmet demand, and ending with zero inventory. Production is subject to monthly capacity limits, and setup costs are incurred only if production occurs in a month.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) lot-sizing problem with fixed setup costs (Capacitated Lot Sizing Problem).
3.  **Define Index Sets:** The primary index is Months, denoted as \( t = 1, \ldots, 24 \), corresponding to the 'Month' column in the CSV.
4.  **Define Decision Variables:**
    -   `x[t]` = Production quantity in month \( t \). Type: GRB.CONTINUOUS (nonnegative real).
    -   `I[t]` = Inventory at the end of month \( t \). Type: GRB.CONTINUOUS (nonnegative real).
    -   `y[t]` = 1 if production is set up in month \( t \), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Production cost per unit: 'ProductionCost' column.
    -   Fixed setup cost per month: 'SetupCost' column.
    -   Inventory holding cost per unit: 'HoldingCost' column.
    -   Monthly demand: 'Demand' column.
    -   Monthly production capacity: 'ProductionCapacity' column.
6.  **Formulate Objective:** Minimize the total cost over all months, which is the sum over all months of (production cost per unit × production quantity) + (fixed setup cost × setup binary) + (inventory holding cost per unit × ending inventory).
7.  **Formulate Constraints:**
    -   Inventory Balance: For each month \( t \), ensure that starting inventory plus production equals demand plus ending inventory. For month 1, initial inventory is zero. For months 2–24, starting inventory is previous month's ending inventory.
    -   Demand Satisfaction: For each month, all demand must be met (no backorders or unmet demand).
    -   Production Capacity and Setup Linking: For each month \( t \), production quantity cannot exceed the month's production capacity times the setup binary (i.e., \( x[t] \leq \text{ProductionCapacity}[t] \cdot y[t] \)), so setup cost is incurred only if production occurs.
    -   Nonnegativity: Production quantities and inventories are nonnegative.
    -   Binary Setup: Setup variables are binary (0 or 1).
    -   Initial Inventory: Inventory at the end of month 0 (before planning starts) is zero.
    -   Final Inventory: Inventory at the end of month 24 must be zero.
[Abstract Model Plan END]
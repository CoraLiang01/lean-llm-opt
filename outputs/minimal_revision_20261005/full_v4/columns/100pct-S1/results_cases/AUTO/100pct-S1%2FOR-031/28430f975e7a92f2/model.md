[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal mix of electricity generation contracts (coal, gas, renewables) to purchase, where each contract is for a fixed-size lot, in order to meet a total electricity demand of 200 units at minimum total cost. Orders must be in whole lots, and each contract option has its own generation per lot and cost per lot as specified in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer lot selection, cost minimization, covering constraint).
3.  **Define Index Sets:** The primary index is the set of available generation contract options, indexed by `i` (each row in energy.csv, representing a specific contract/lot offer).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of lots to purchase of contract option `i`. Type: GRB.INTEGER (must be whole lots; can be zero).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'cost_per_lot' (the cost to purchase one lot of option `i`).
    -   Constraint coefficients: 'gen_per_lot' (the amount of generation provided by one lot of option `i`).
    -   Constraint RHS: The total demand to be met, which is 200 (given in the query).
    -   Technology type: 'tech' (categorizes each option as coal, gas, or renewables; used for reporting or further constraints if needed).
6.  **Formulate Objective:** Minimize the total procurement cost, i.e., minimize sum over all options of (`cost_per_lot[i]` * `x[i]`).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): The total generation from all purchased lots must be at least 200 units: sum over all options of (`gen_per_lot[i]` * `x[i]`) ≥ 200.
    -   Constraint 2 (Integrality): For all options `i`, `x[i]` ≥ 0 and integer (cannot purchase fractional lots; can purchase zero lots of any option).
    -   (No further constraints are specified in the query, such as technology minimums/maximums or emissions, so only the above are required.)
[Abstract Model Plan END]
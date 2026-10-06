[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal mix of electricity generation contracts (coal, gas, renewables) to purchase, where each contract is for a fixed-size lot, in order to meet a total electricity demand of 200 units at minimum total cost. Orders must be in whole lots, and each contract option has its own generation per lot and cost per lot as specified in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer lot selection, cost minimization, covering constraint).
3.  **Define Index Sets:** The primary index is the set of available generation contract options, indexed by `i` (each row in energy.csv, representing a unique contract/option).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of lots to purchase of contract option `i`. Type: GRB.INTEGER (must be whole lots, can be zero).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: `cost_per_lot` (float, cost to purchase one lot of option `i`).
    -   Constraint coefficients: `gen_per_lot` (int, generation provided by one lot of option `i`).
    -   Constraint RHS: Total demand = 200 (given in the query, not in the CSV).
    -   Technology type: `tech` (categorical, identifies if option is coal, gas, or renewables; used for reporting or further constraints if needed).
6.  **Formulate Objective:** Minimize the total procurement cost, i.e., minimize the sum over all options of (`cost_per_lot[i]` * `x[i]`).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): The total generation from all purchased lots must be at least 200 units: sum over all options of (`gen_per_lot[i]` * `x[i]`) ≥ 200.
    -   Constraint 2 (Integrality): For all options `i`, `x[i]` ≥ 0 and integer (whole lots only).
    -   (No explicit upper bound on lots per option unless specified elsewhere; all 131 options are available for selection.)
[Abstract Model Plan END]
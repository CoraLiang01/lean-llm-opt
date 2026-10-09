[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal number of whole-lot purchases for each available generation contract (coal, gas, renewables) to meet a fixed total electricity demand (200 units), while minimizing total procurement cost. Each contract option has a fixed generation per lot and cost per lot, and only whole lots can be purchased.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer lot sizes, cost minimization, single-period resource allocation).
3.  **Define Index Sets:** The primary index is the set of available generation contract options, denoted as `i ∈ Options` (where each option is a row in energy.csv, uniquely identified by the 'option' column).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of lots to purchase for contract option `i`. Type: GRB.INTEGER (must be whole lots, ≥ 0).
5.  **Identify Parameters (from Schema):**
    -   Generation per lot: from column `'gen_per_lot'` (integer, units of generation per lot for option `i`).
    -   Cost per lot: from column `'cost_per_lot'` (float, cost per lot for option `i`).
    -   Technology type: from column `'tech'` (categorical: 'coal', 'gas', 'renewables'; used for reporting or possible future constraints, but not directly needed for this model as all options are eligible).
    -   Total demand: fixed at 200 (from query, not schema).
6.  **Formulate Objective:** Minimize total procurement cost, i.e., minimize the sum over all options of (cost per lot) × (number of lots purchased):  
    Minimize:  sum over i of  `cost_per_lot[i] * x[i]`
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): The total generation from all purchased lots must be at least the required demand (200 units):  
        sum over i of `gen_per_lot[i] * x[i]`  ≥  200
    -   Constraint 2 (Integrality): Each `x[i]` must be an integer ≥ 0 (whole lots only).
    -   (No upper bound on lots per option unless specified elsewhere; all options in the CSV are eligible.)
[Abstract Model Plan END]
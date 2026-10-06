[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal number of whole-lot purchases for each available generation contract (coal, gas, renewables) to meet a fixed total electricity demand (200 units), while minimizing total procurement cost. Each contract option has a fixed generation per lot and a cost per lot, and only whole lots can be purchased.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (specifically, an integer lot-sizing/blending problem).
3.  **Define Index Sets:** The primary index is the set of available generation contract options, denoted as `i ∈ Options` (where each option is a row in energy.csv, e.g., coal_001, gas_002, renewables_003, etc.).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of lots of contract option `i` to purchase. Type: GRB.INTEGER (must be whole lots, i.e., non-negative integers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'cost_per_lot' (the cost to purchase one lot of option `i`).
    -   Constraint coefficients: 'gen_per_lot' (the amount of electricity generated per lot for option `i`).
    -   Constraint RHS: The total demand to be met, which is 200 (as specified in the query).
6.  **Formulate Objective:** Minimize the total procurement cost, i.e., minimize the sum over all options of (`cost_per_lot[i]` * `x[i]`).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): The total generation from all purchased lots must be at least 200 units. That is, sum over all options of (`gen_per_lot[i]` * `x[i]`) ≥ 200.
    -   Constraint 2 (Integrality): For all options `i`, `x[i]` ≥ 0 and integer (whole lots only).
    -   (No explicit upper bound on lots per option unless specified elsewhere; all 131 options are available for selection.)
[Abstract Model Plan END]
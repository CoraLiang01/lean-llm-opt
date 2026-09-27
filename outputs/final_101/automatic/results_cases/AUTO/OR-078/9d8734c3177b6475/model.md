[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal mix of electricity generation contracts (coal, gas, renewables) to purchase in whole lots, such that total generation meets a fixed demand (200 units), and total cost is minimized. Each contract option specifies a technology, a fixed generation per lot, and a cost per lot.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer lot selection, cost minimization, single-period resource allocation).
3.  **Define Index Sets:** The primary index is the set of available generation contract options, denoted as `i ∈ Options` (where each option is a row in energy.csv, uniquely identified by the 'option' column).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of lots of contract option `i` to purchase. Type: GRB.INTEGER (must be whole lots, can be zero or more).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'cost_per_lot' (total cost for each lot of option `i`).
    -   Generation per lot: 'gen_per_lot' (amount of electricity provided by one lot of option `i`).
    -   Technology type: 'tech' (categorical, for reporting or further constraints if needed).
    -   Demand (RHS): Fixed value, 200 (from query, not from CSV).
6.  **Formulate Objective:** Minimize the total procurement cost, i.e., minimize sum over all options of (`cost_per_lot[i]` * `x[i]`).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): The total generation from all selected lots must be at least 200 units: sum over all options of (`gen_per_lot[i]` * `x[i]`) ≥ 200.
    -   Constraint 2 (Integrality): For all options `i`, `x[i]` ∈ {0, 1, 2, ...} (integer, non-negative).
    -   (No explicit upper bound on lots per option unless specified elsewhere; all 131 options are available for selection.)
    -   (No minimum/maximum per technology unless further specified; all options are eligible.)
[Abstract Model Plan END]
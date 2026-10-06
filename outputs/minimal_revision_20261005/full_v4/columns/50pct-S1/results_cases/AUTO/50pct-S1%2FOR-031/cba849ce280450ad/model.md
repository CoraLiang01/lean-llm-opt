[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal selection of generation contracts (coal, gas, renewables) to meet a fixed electricity demand (200 units), where each contract must be purchased in whole lots, and each lot provides a fixed amount of generation at a specified cost. The goal is to minimize total procurement cost.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (specifically, an integer lot-sizing and blending problem).
3.  **Define Index Sets:** The primary index is the set of available generation contract options, indexed by `i` (where each `i` corresponds to a row in energy.csv, i.e., a specific contract/lot).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of lots of contract option `i` to purchase. Type: GRB.INTEGER (must be non-negative integers, as only whole lots can be purchased).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'cost_per_lot' (the cost to purchase one lot of option `i`).
    -   Constraint coefficients: 'gen_per_lot' (the amount of generation provided by one lot of option `i`).
    -   Constraint RHS: The total demand to be met, which is 200 (as specified in the query).
6.  **Formulate Objective:** Minimize the total cost of purchased lots, i.e., minimize sum over all options of (`cost_per_lot[i]` * `x[i]`).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): The total generation from all purchased lots must be at least 200 units: sum over all options of (`gen_per_lot[i]` * `x[i]`) ≥ 200.
    -   Constraint 2 (Lot Integrality): For all options `i`, `x[i]` ≥ 0 and integer (only whole lots can be purchased).
    -   (No additional constraints are specified in the query; all options are available for selection, and there are no upper bounds or technology-specific requirements unless further specified.)
[Abstract Model Plan END]
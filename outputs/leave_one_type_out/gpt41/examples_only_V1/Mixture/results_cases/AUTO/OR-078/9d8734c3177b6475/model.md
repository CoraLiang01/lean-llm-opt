[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal mix of electricity generation contracts (coal, gas, renewables) to purchase, where each contract is for a fixed-size "lot" (integer multiples only), in order to meet a total electricity demand of 200 units at minimum total cost. All contract options are listed in energy.csv, and each lot has a specified generation amount and cost.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (specifically, an integer lot-sizing and blending problem).
3.  **Define Index Sets:** The primary index is the set of available generation contract options, indexed by `i` (where each `i` corresponds to a row in energy.csv, i.e., a specific contract/lot offer).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of lots to purchase of contract option `i`. Type: GRB.INTEGER (must be integer, as lots cannot be split).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'cost_per_lot' (the cost to purchase one lot of option `i`).
    -   Constraint coefficients: 'gen_per_lot' (the amount of generation provided by one lot of option `i`).
    -   Constraint RHS: The total demand to be met, which is 200 (from the query).
6.  **Formulate Objective:** Minimize the total procurement cost, i.e., minimize the sum over all options of (cost_per_lot[i] * x[i]).
7.  **Formulate Constraints:**
    -   Demand Satisfaction: The sum over all options of (gen_per_lot[i] * x[i]) must be at least 200 (i.e., total purchased generation meets or exceeds demand).
    -   Lot Integrality: For all options `i`, x[i] must be a non-negative integer (x[i] ≥ 0 and integer).
    -   (No additional constraints are specified in the query; all options in the CSV are available for selection, and there are no upper bounds on lots per option unless specified in the data.)
[Abstract Model Plan END]
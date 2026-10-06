[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal number of whole-lot purchases for each available generation contract (coal, gas, renewables) to meet a fixed total electricity demand (200 units), while minimizing total procurement cost. Each contract option has a fixed generation per lot and a cost per lot, and only whole lots can be purchased.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (specifically, an integer lot-sizing/blending problem).
3.  **Define Index Sets:** The primary index is the set of available generation contract options, denoted as `i ∈ Options`, where each option corresponds to a row in the CSV (option: coal_001, gas_001, renewables_001, etc.).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of lots to purchase for contract option `i`. Type: GRB.INTEGER (must be whole lots, ≥ 0).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: `cost_per_lot` (the cost to purchase one lot of option `i`).
    -   Constraint coefficients: `gen_per_lot` (the amount of generation provided by one lot of option `i`).
    -   Constraint RHS: The total demand to be met, which is 200 (given in the query, not in the CSV).
6.  **Formulate Objective:** Minimize the total procurement cost, i.e., minimize the sum over all options of (`cost_per_lot[i]` * `x[i]`).
7.  **Formulate Constraints:**
    -   Demand Satisfaction: The sum over all options of (`gen_per_lot[i]` * `x[i]`) must be greater than or equal to 200 (i.e., total purchased generation meets or exceeds demand).
    -   Integrality: For all options `i`, `x[i]` must be integer and ≥ 0 (cannot purchase fractional lots or negative lots).
    -   (No additional constraints are specified in the query; all options are available for selection, and there are no upper bounds or technology-specific requirements unless further specified.)
[Abstract Model Plan END]
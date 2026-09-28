[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal mix of electricity generation contracts (coal, gas, renewables) to purchase, where each contract is for a fixed-size "lot" (integer multiples only), in order to meet a total electricity demand of 200 units at minimum total cost. All available contract options are listed in energy.csv.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer lot selection, cost minimization, covering constraint).
3.  **Define Index Sets:** The primary index is the set of all generation contract options, indexed by `i` (where each `i` corresponds to a row in energy.csv, i.e., a specific contract/option).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of lots to purchase of contract option `i`. Type: GRB.INTEGER (must be integer, as lots cannot be split).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'cost_per_lot' (the cost to purchase one lot of option `i`).
    -   Generation per lot: 'gen_per_lot' (the amount of electricity provided by one lot of option `i`).
    -   Technology type: 'tech' (categorical, e.g., coal, gas, renewables; may be used for reporting or further constraints if needed).
    -   The total demand to be met: 200 (given in the query, not in the CSV).
6.  **Formulate Objective:** Minimize the total procurement cost, i.e., minimize sum over all options of (`cost_per_lot[i]` * `x[i]`).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): The total electricity generated from all purchased lots must be at least 200 units: sum over all options of (`gen_per_lot[i]` * `x[i]`) ≥ 200.
    -   Constraint 2 (Integrality): For all options `i`, `x[i]` ≥ 0 and integer (cannot purchase negative or fractional lots).
    -   (No further constraints are specified in the query; all options in the CSV are available for selection, and there are no explicit technology minimums/maximums or other operational constraints.)
[Abstract Model Plan END]
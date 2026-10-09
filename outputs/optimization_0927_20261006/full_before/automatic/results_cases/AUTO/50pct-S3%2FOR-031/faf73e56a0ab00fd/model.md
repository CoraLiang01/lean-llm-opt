[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal number of lots to purchase from each available generation contract (coal, gas, renewables) to meet a fixed electricity demand (200 units), while minimizing total procurement cost. Each contract option has a fixed generation per lot and a cost per lot, and orders must be in whole lots.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer lot decisions, linear objective and constraints).
3.  **Define Index Sets:** The primary index is the set of available generation contract options, denoted as `i ∈ Options`, where each option is a row in energy.csv (option: coal_001, gas_002, renewables_003, etc.).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of lots to purchase for contract option `i`. Type: GRB.INTEGER (must be whole lots, ≥ 0).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: `cost_per_lot` (cost to purchase one lot of option `i`).
    -   Constraint coefficients: `gen_per_lot` (generation provided by one lot of option `i`).
    -   Constraint RHS: Total demand = 200 (given in the query).
    -   Option type: `tech` (coal, gas, renewables) — used for reporting or further constraints if needed.
6.  **Formulate Objective:** Minimize total procurement cost, i.e., minimize the sum over all options of (`cost_per_lot[i]` * `x[i]`).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): The total generation from all purchased lots must meet or exceed the required demand: sum over all options of (`gen_per_lot[i]` * `x[i]`) ≥ 200.
    -   Constraint 2 (Integrality): For all options `i`, `x[i]` must be integer and ≥ 0 (cannot purchase negative or fractional lots).
    -   (No further constraints are specified in the query; all options in the CSV are available for selection.)
[Abstract Model Plan END]
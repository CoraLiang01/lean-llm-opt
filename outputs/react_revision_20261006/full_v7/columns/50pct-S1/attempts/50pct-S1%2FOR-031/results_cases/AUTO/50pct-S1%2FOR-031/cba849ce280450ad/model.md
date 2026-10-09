[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal selection of generation contracts (lots) from coal, gas, and renewables to meet a total electricity demand of 200 units, while minimizing total procurement costs. Each contract (option) must be purchased in whole lots, and each lot provides a fixed amount of generation at a specified cost.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer lot selection, cost minimization, covering constraint).
3.  **Define Index Sets:** The primary index is the set of available generation contract options, denoted as `i ∈ Options` (each row in energy.csv, uniquely identified by 'option').
4.  **Define Decision Variables:**
    -   `x[i]` = Number of lots of contract option `i` to purchase. Type: GRB.INTEGER (must be whole lots, ≥ 0).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'cost_per_lot' (cost to purchase one lot of option `i`).
    -   Constraint coefficients: 'gen_per_lot' (generation provided by one lot of option `i`).
    -   Constraint RHS: Total demand = 200 (given in the query).
    -   Option type: 'tech' (coal, gas, renewables) — used for reporting or further constraints if needed, but not directly in this model.
6.  **Formulate Objective:** Minimize the total procurement cost, i.e., minimize the sum over all options of (cost_per_lot[i] * x[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): The total generation from all selected lots must meet or exceed the required demand: sum over all options of (gen_per_lot[i] * x[i]) ≥ 200.
    -   Constraint 2 (Integrality and Non-negativity): For all options `i`, x[i] ∈ {0, 1, 2, ...} (integer, ≥ 0).
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal mix of electricity generation contracts (coal, gas, renewables) to purchase, where each contract is for a fixed-size lot, in order to meet a total electricity demand of 200 units at minimum total cost. Orders must be in whole lots, and all contract options are available as described in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer lot selection, cost minimization, covering constraint).
3.  **Define Index Sets:** The primary index is the set of available generation contract options, indexed by \( i \) (each row in energy.csv, representing a unique contract/option).
4.  **Define Decision Variables:**
    -   \( x[i] \) = Number of lots to purchase of contract option \( i \). Type: GRB.INTEGER (must be whole lots, can be zero).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'cost_per_lot' (cost to purchase one lot of option \( i \)).
    -   Constraint coefficients: 'gen_per_lot' (generation provided by one lot of option \( i \)).
    -   Constraint RHS: Total demand = 200 (fixed value from query, not from schema).
6.  **Formulate Objective:** Minimize total procurement cost, i.e., minimize \(\sum_{i} \text{cost\_per\_lot}[i] \times x[i]\).
7.  **Formulate Constraints:**
    -   Demand Satisfaction: \(\sum_{i} \text{gen\_per\_lot}[i] \times x[i] \geq 200\) (total generation from all purchased lots must meet or exceed demand).
    -   Integrality: \(x[i] \geq 0\), integer, for all \(i\) (cannot purchase negative or fractional lots).
[Abstract Model Plan END]
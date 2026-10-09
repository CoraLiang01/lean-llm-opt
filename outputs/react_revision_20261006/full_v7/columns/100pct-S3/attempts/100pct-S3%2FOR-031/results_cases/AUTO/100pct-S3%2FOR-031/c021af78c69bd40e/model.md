[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal mix of electricity generation contracts (coal, gas, renewables) to purchase, in whole lots, to meet a total demand of 200 units at minimum total cost. Each contract option specifies a fixed generation per lot and a cost per lot. Orders must be in integer lots.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer lot sizes, cost minimization, covering/blending).
3.  **Define Index Sets:** The primary index is the set of available generation contract options, denoted as \( i \in \text{Options} \), where each option corresponds to a row in energy.csv and is associated with a technology type (coal, gas, renewables).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of lots to purchase of contract option \( i \). Type: GRB.INTEGER (must be non-negative and integer, as lots cannot be split).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'cost_per_lot' (the cost to purchase one lot of option \( i \)).
    -   Constraint coefficients: 'gen_per_lot' (the amount of generation provided by one lot of option \( i \)).
    -   Constraint RHS: The total demand to be met, which is 200 (given in the query).
    -   Option identifiers: 'option' (unique contract code), 'tech' (technology type: coal, gas, renewables).
6.  **Formulate Objective:** Minimize the total procurement cost, i.e., minimize the sum over all options of (cost_per_lot[i] * x[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): The total generation from all purchased lots must be at least 200 units. That is, sum over all options of (gen_per_lot[i] * x[i]) ≥ 200.
    -   Constraint 2 (Integrality and Non-negativity): For all options \( i \), x[i] ≥ 0 and integer.
    -   (No further constraints are specified in the query; all contract options in the CSV are available for selection, and there are no explicit technology minimums, maximums, or exclusivity requirements.)
[Abstract Model Plan END]
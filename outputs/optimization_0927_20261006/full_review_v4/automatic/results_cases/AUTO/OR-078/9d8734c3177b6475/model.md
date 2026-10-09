[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal mix of electricity generation contracts (coal, gas, renewables) to purchase, where each contract is for a fixed-size lot, in order to meet a total electricity demand of 200 units at minimum total cost. Orders must be in whole lots, and all available contract options are listed in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer lot selection, cost minimization, covering constraint).
3.  **Define Index Sets:** The primary index is the set of generation contract options \( i \) (from all rows in energy.csv), each uniquely identified by the 'option' field.
4.  **Define Decision Variables:**
    -   \( x[i] \) = Number of lots to purchase of contract option \( i \). Type: GRB.INTEGER (non-negative, whole lots only).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'cost_per_lot' (total cost for each lot of option \( i \)).
    -   Constraint coefficients: 'gen_per_lot' (generation provided per lot for option \( i \)).
    -   Constraint RHS: Total demand = 200 (fixed value from query, not from schema).
6.  **Formulate Objective:** Minimize the total procurement cost, i.e., minimize \(\sum_{i} \text{cost\_per\_lot}[i] \cdot x[i]\).
7.  **Formulate Constraints:**
    -   Demand Satisfaction: \(\sum_{i} \text{gen\_per\_lot}[i] \cdot x[i] \geq 200\) (total generation from all selected lots must meet or exceed demand).
    -   Integrality: \(x[i] \in \mathbb{Z}_{\geq 0}\) for all \(i\) (must purchase whole lots, zero or more of each option).
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal mix of electricity generation contracts (coal, gas, renewables) to purchase, where each contract is for a fixed-size lot, in order to meet a total electricity demand of 200 units at minimum total cost. Orders must be in whole lots, and all available contract options are listed in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer lot selection, cost minimization, covering constraint).
3.  **Define Index Sets:** The primary index is the set of available generation contract options, indexed by \( i \) (i.e., all rows in energy.csv, each representing a unique contract/lot).
4.  **Define Decision Variables:**
    -   \( x[i] \) = Number of lots of contract option \( i \) to purchase. Type: GRB.INTEGER (non-negative, whole lots only).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'cost_per_lot' (total cost for each lot option).
    -   Constraint coefficients: 'gen_per_lot' (generation provided per lot for each option).
    -   Constraint RHS: Total demand to be met (200 units, as specified in the query).
6.  **Formulate Objective:** Minimize the total procurement cost, i.e., minimize \(\sum_{i} \text{cost\_per\_lot}[i] \cdot x[i]\).
7.  **Formulate Constraints:**
    -   Demand Satisfaction: The total generation from all selected lots must meet or exceed the required demand, i.e., \(\sum_{i} \text{gen\_per\_lot}[i] \cdot x[i] \geq 200\).
    -   Integrality and Non-negativity: For all \( i \), \( x[i] \) are integer variables with \( x[i] \geq 0 \).
[Abstract Model Plan END]
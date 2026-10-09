[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal number of whole-lot purchases for each available generation contract (coal, gas, renewables) to meet a fixed total electricity demand (200 units), while minimizing total procurement cost. Each contract option has a fixed generation per lot and cost per lot, and orders must be in integer lots.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (integer lot sizes, cost minimization, covering constraint).
3.  **Define Index Sets:** The primary index is the set of available generation contract options, indexed by \( i \) (where each \( i \) corresponds to a row in energy.csv, representing a specific contract for coal, gas, or renewables).
4.  **Define Decision Variables:**
    -   \( x[i] \) = Number of lots to purchase for contract option \( i \). Type: GRB.INTEGER (non-negative, whole lots only).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'cost_per_lot' (cost to purchase one lot of contract \( i \)).
    -   Constraint coefficients: 'gen_per_lot' (generation provided by one lot of contract \( i \)).
    -   Constraint RHS: Total demand (fixed at 200, as specified in the query).
6.  **Formulate Objective:** Minimize the total procurement cost, i.e., minimize the sum over all contract options of ('cost_per_lot' * \( x[i] \)).
7.  **Formulate Constraints:**
    -   Demand Satisfaction: The sum over all contract options of ('gen_per_lot' * \( x[i] \)) must be greater than or equal to 200 (i.e., total purchased generation meets or exceeds demand).
    -   Integrality and Non-negativity: For all \( i \), \( x[i] \) must be integer and \( x[i] \geq 0 \).
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal number of lots to purchase from each available generation contract (coal, gas, renewables) to meet a total electricity demand of 200 units, while minimizing total procurement cost. Each contract (row in the CSV) specifies a technology type, a fixed generation amount per lot, and a cost per lot. Orders must be in whole lots (integer multiples).
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer lot decisions, linear objective and constraints).
3.  **Define Index Sets:** The primary index is the set of available generation contract options, indexed by \( i \) (each row in energy.csv, uniquely identified by 'option').
4.  **Define Decision Variables:**
    -   \( x[i] \) = Number of lots to purchase for contract option \( i \). Type: GRB.INTEGER (non-negative, whole lots).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'cost_per_lot' (cost to purchase one lot of option \( i \)).
    -   Constraint coefficients: 'gen_per_lot' (generation provided by one lot of option \( i \)).
    -   Constraint RHS: Total demand (fixed at 200, as specified in the query).
    -   Indexing: 'option' (unique contract identifier), 'tech' (technology type: coal, gas, renewables).
6.  **Formulate Objective:** Minimize the total procurement cost, i.e., minimize \(\sum_{i} \text{cost\_per\_lot}[i] \times x[i]\), summing over all contract options \( i \).
7.  **Formulate Constraints:**
    -   Demand Satisfaction: Ensure total generation meets or exceeds demand: \(\sum_{i} \text{gen\_per\_lot}[i] \times x[i] \geq 200\).
    -   Integrality and Non-negativity: For all \( i \), \( x[i] \) are integer and \( x[i] \geq 0 \).
[Abstract Model Plan END]
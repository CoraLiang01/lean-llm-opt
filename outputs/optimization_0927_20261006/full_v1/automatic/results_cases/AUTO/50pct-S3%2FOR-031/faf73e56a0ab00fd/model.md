[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal number of lots to purchase from each available generation contract (coal, gas, renewables) to meet a fixed electricity demand (200 units), while minimizing total procurement cost. Each contract option has a fixed generation per lot and cost per lot, and orders must be in whole lots.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer lot sizes, linear objective and constraints).
3.  **Define Index Sets:** The primary index is the set of available generation contract options, denoted as \( i \in \text{Options} \), where each option corresponds to a row in energy.csv.
4.  **Define Decision Variables:**
    -   \( x[i] \) = Number of lots to purchase for contract option \( i \). Type: GRB.INTEGER (must be non-negative and integral).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'cost_per_lot' (cost to purchase one lot of option \( i \)).
    -   Constraint coefficients: 'gen_per_lot' (generation provided by one lot of option \( i \)).
    -   Constraint RHS: Total demand (fixed at 200, as specified in the query).
6.  **Formulate Objective:** Minimize the total procurement cost, i.e., minimize \( \sum_{i \in \text{Options}} \text{cost\_per\_lot}[i] \times x[i] \).
7.  **Formulate Constraints:**
    -   Demand Satisfaction: \( \sum_{i \in \text{Options}} \text{gen\_per\_lot}[i] \times x[i] \geq 200 \) (total scheduled generation must meet or exceed demand).
    -   Integrality and Non-negativity: \( x[i] \in \mathbb{Z}_{\geq 0} \) for all \( i \in \text{Options} \) (must purchase a non-negative integer number of lots for each contract option).
[Abstract Model Plan END]
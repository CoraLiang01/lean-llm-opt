[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal mix of electricity generation contracts (coal, gas, renewables) to purchase in whole lots, such that the total generation meets exactly a specified demand (200 units), while minimizing total procurement cost. Each contract option provides a fixed generation per lot and has a specified cost per lot.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer lot selection, linear objective and constraints).
3.  **Define Index Sets:** The primary index is the set of available generation contract options, denoted as \( i \in \text{Options} \), where each option is a row in energy.csv (option, tech, gen_per_lot, cost_per_lot, unit_cost_est).
4.  **Define Decision Variables:**
    -   \( x[i] \) = Number of lots to purchase for contract option \( i \). Type: GRB.INTEGER, \( x[i] \geq 0 \).
5.  **Identify Parameters (from Schema):**
    -   Generation per lot: energy.csv column 'gen_per_lot' (numeric, units consistent with demand).
    -   Cost per lot: energy.csv column 'cost_per_lot' (numeric, used in objective).
    -   Technology type: energy.csv column 'tech' (categorical, for reporting or further constraints if needed).
    -   Demand: Fixed value 200 (from query, not schema).
6.  **Formulate Objective:** Minimize total procurement cost, i.e., minimize \( \sum_{i \in \text{Options}} \text{cost\_per\_lot}[i] \times x[i] \).
7.  **Formulate Constraints:**
    -   Demand Satisfaction: \( \sum_{i \in \text{Options}} \text{gen\_per\_lot}[i] \times x[i] = 200 \) (total generation exactly meets demand).
    -   Integrality and Non-negativity: \( x[i] \in \mathbb{Z}_{\geq 0} \) for all \( i \) (lots must be whole and non-negative).
[Abstract Model Plan END]
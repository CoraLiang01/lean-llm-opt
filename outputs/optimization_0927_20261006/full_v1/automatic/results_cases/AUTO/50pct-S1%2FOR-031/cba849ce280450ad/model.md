[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal selection of generation contracts (lots) from coal, gas, and renewables to meet a fixed electricity demand (200 units), where each contract must be purchased in whole lots, and the goal is to minimize total procurement cost.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (integer lot selection, cost minimization, single-period resource allocation).
3.  **Define Index Sets:** The primary index is the set of available generation options (contracts), indexed by \( i \), as listed in the 'option' column of energy.csv. Each option is associated with a technology type (coal, gas, renewables).
4.  **Define Decision Variables:**
    -   \( x[i] \) = Number of lots of generation option \( i \) to purchase. Type: GRB.INTEGER (must be whole lots, \( x[i] \geq 0 \)).
5.  **Identify Parameters (from Schema):**
    -   Generation per lot: schema['gen_per_lot'][i] (amount of electricity provided by one lot of option \( i \)).
    -   Cost per lot: schema['cost_per_lot'][i] (cost to purchase one lot of option \( i \)).
    -   Technology type: schema['tech'][i] (categorical, for reporting or further constraints if needed).
    -   Demand: Fixed value (200), provided in the query.
6.  **Formulate Objective:** Minimize total procurement cost, i.e., minimize \( \sum_{i} \text{cost\_per\_lot}[i] \times x[i] \).
7.  **Formulate Constraints:**
    -   Demand Satisfaction: \( \sum_{i} \text{gen\_per\_lot}[i] \times x[i] \geq 200 \) (total purchased generation must meet or exceed demand).
    -   Integrality and Non-negativity: \( x[i] \in \mathbb{Z}_{\geq 0} \) for all \( i \) (lots must be whole and non-negative).
[Abstract Model Plan END]
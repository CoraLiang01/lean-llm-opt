[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal allocation of goods from multiple distribution centers to multiple customer groups, such that all customer demands are satisfied, no distribution center exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Distribution Centers (Sources): indexed by $s$ (from supply_capacity.csv and transportation_costs.csv, e.g., S1, S2, ..., S12)
    - Customer Groups (Destinations): indexed by $c$ (from customer_demand.csv and transportation_costs.csv, e.g., C1, C2, ..., C12)
4.  **Define Decision Variables:**
    -   $x_{s,c}$ = quantity of goods shipped from distribution center $s$ to customer group $c$. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from $s$ to $c$: from 'transportation_costs.csv', columns C1–C12 for each row S1–S12.
    -   Supply capacity at each distribution center $s$: from 'supply_capacity.csv', column 'supply_capacity'.
    -   Demand for each customer group $c$: from 'customer_demand.csv', column 'demand'.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize $\sum_{s}\sum_{c} \text{transportation\_cost}_{s,c} \cdot x_{s,c}$.
7.  **Formulate Constraints:**
    -   Supply Capacity Constraints: For each distribution center $s$, $\sum_{c} x_{s,c} \leq \text{supply\_capacity}_s$ (do not exceed supply at any center).
    -   Demand Satisfaction Constraints: For each customer group $c$, $\sum_{s} x_{s,c} = \text{demand}_c$ (fully meet each customer group's demand).
    -   Non-negativity: For all $s, c$, $x_{s,c} \geq 0$.
[Abstract Model Plan END]
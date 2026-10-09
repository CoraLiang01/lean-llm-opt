[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan for delivering goods from Walmart stores (suppliers) to customer groups, such that all customer demands are met, no store exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Stores (sources): indexed by $s$ (from supply_capacity.csv and transportation_costs.csv, e.g., S1, S2, ..., S11)
    - Customers (destinations): indexed by $c$ (from customer_demand.csv and transportation_costs.csv, e.g., C1, C2, ..., C12)
4.  **Define Decision Variables:**
    -   $x_{s,c}$ = Quantity of goods transported from store $s$ to customer $c$. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from store $s$ to customer $c$: from 'transportation_costs.csv', columns C1–C12 for each row S1–S11.
    -   Supply capacity of each store $s$: from 'supply_capacity.csv', column 'supply_capacity'.
    -   Demand of each customer $c$: from 'customer_demand.csv', column 'demand'.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all stores and customers of (transportation cost per unit from $s$ to $c$) times ($x_{s,c}$).
7.  **Formulate Constraints:**
    -   Supply Capacity Constraint (for each store $s$): The total quantity shipped from store $s$ to all customers cannot exceed its supply capacity. $\sum_{c} x_{s,c} \leq$ supply_capacity[$s$].
    -   Demand Satisfaction Constraint (for each customer $c$): The total quantity received by customer $c$ from all stores must exactly meet its demand. $\sum_{s} x_{s,c} = $ demand[$c$].
    -   Non-negativity: $x_{s,c} \geq 0$ for all $s, c$.
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily transportation plan—specifically, how much to ship from each distribution center to each customer group—so that all customer demands are satisfied, no distribution center exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Distribution Centers (sources): indexed by $s$ (from 'supply_capacity.csv', column 'Unnamed: 0', e.g., S1, S2, ..., S18)
    - Customer Groups (destinations): indexed by $c$ (from 'customer_demand.csv', column 'customer', e.g., C1, C2, ..., C18)
4.  **Define Decision Variables:**
    - $x_{s,c}$ = quantity of goods transported from distribution center $s$ to customer group $c$. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    - Transportation cost per unit from $s$ to $c$: from 'transportation_costs.csv', columns 'C1'...'C18' (rows indexed by 'Unnamed: 0' = $s$).
    - Supply capacity at each distribution center $s$: from 'supply_capacity.csv', column 'supply_capacity'.
    - Demand for each customer group $c$: from 'customer_demand.csv', column 'demand'.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize $\sum_{s}\sum_{c} \text{transportation\_cost}_{s,c} \cdot x_{s,c}$, where $\text{transportation\_cost}_{s,c}$ is the per-unit cost from $s$ to $c$.
7.  **Formulate Constraints:**
    - Supply Capacity Constraint: For each distribution center $s$, the total quantity shipped out cannot exceed its supply capacity: $\sum_{c} x_{s,c} \leq \text{supply\_capacity}_s$.
    - Demand Satisfaction Constraint: For each customer group $c$, the total quantity received must exactly meet its demand: $\sum_{s} x_{s,c} = \text{demand}_c$.
    - Non-negativity: For all $s, c$, $x_{s,c} \geq 0$.
[Abstract Model Plan END]
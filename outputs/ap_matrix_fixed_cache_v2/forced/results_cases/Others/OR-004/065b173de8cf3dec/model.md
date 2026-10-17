[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal allocation of goods from multiple distribution centers to multiple customer groups, such that all customer demands are satisfied, no distribution center exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Distribution centers (sources), indexed by \( s \) (from 'supply_capacity.csv', field 'Unnamed: 0', e.g., S1, S2, ..., S12)
    - Customer groups (destinations), indexed by \( c \) (from 'customer_demand.csv', field 'customer', e.g., C1, C2, ..., C12)
4.  **Define Decision Variables:**
    -   \( x_{s,c} \) = quantity of goods shipped from distribution center \( s \) to customer group \( c \). Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from \( s \) to \( c \): from 'transportation_costs.csv', columns 'C1'...'C12' for each row 'Unnamed: 0' = \( s \).
    -   Supply capacity at each distribution center \( s \): from 'supply_capacity.csv', column 'supply_capacity'.
    -   Demand for each customer group \( c \): from 'customer_demand.csv', column 'demand'.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all \( s \) and \( c \) of (transportation cost per unit from \( s \) to \( c \)) times \( x_{s,c} \).
7.  **Formulate Constraints:**
    -   Supply Capacity Constraint: For each distribution center \( s \), the total quantity shipped out does not exceed its supply capacity: \(\sum_{c} x_{s,c} \leq \text{supply\_capacity}[s]\).
    -   Demand Satisfaction Constraint: For each customer group \( c \), the total quantity received from all distribution centers meets its demand exactly: \(\sum_{s} x_{s,c} = \text{demand}[c]\).
    -   Non-negativity Constraint: For all \( s, c \), \( x_{s,c} \geq 0 \).
[Abstract Model Plan END]
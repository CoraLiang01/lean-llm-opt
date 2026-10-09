[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily shipment quantities from each distribution center (supplier) to each customer group, such that all customer demands are fully met, no distribution center exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Suppliers (distribution centers), indexed by $i$ (from 'Supplier' in supply_capacity.csv and 'Unnamed: 0' in transportation_costs.csv).
    - Customers (customer groups), indexed by $j$ (from 'Customers' in customer_demand.csv and column headers in transportation_costs.csv).
4.  **Define Decision Variables:**
    - $x_{i,j}$ = Quantity of goods shipped from supplier $i$ to customer $j$ per day. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    - Transportation cost per unit from supplier $i$ to customer $j$: from 'transportation_costs.csv', field $c_{i,j}$.
    - Supply capacity of supplier $i$: from 'supply_capacity.csv', field 'supply_capacity'.
    - Demand of customer $j$: from 'customer_demand.csv', field 'demand'.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize $\sum_{i} \sum_{j} c_{i,j} \cdot x_{i,j}$, where $c_{i,j}$ is the cost per unit from supplier $i$ to customer $j$.
7.  **Formulate Constraints:**
    - Constraint 1 (Demand Satisfaction): For each customer $j$, the total goods received from all suppliers must equal their demand: $\sum_{i} x_{i,j} = \text{demand}_j$.
    - Constraint 2 (Supply Capacity): For each supplier $i$, the total goods shipped to all customers must not exceed its supply capacity: $\sum_{j} x_{i,j} \leq \text{supply\_capacity}_i$.
    - Constraint 3 (Non-negativity): $x_{i,j} \geq 0$ for all $i, j$.
[Abstract Model Plan END]
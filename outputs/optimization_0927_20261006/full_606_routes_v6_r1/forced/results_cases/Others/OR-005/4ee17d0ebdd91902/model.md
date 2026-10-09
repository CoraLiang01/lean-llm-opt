[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily shipment quantities from each distribution center (supplier) to each customer group, such that all customer demands are fully met, no distribution center exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Suppliers (distribution centers), indexed by $i$ (from 'supply_capacity.csv', column 'Supplier')
    - Customers (customer groups), indexed by $j$ (from 'customer_demand.csv', column 'Customers')
4.  **Define Decision Variables:**
    -   $x_{i,j}$ = quantity of goods shipped from supplier $i$ to customer $j$ per day. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from supplier $i$ to customer $j$: from 'transportation_costs.csv', entry at row $i$ (matching 'Unnamed: 0' to supplier) and column $j$ (matching customer group).
    -   Supply capacity of each supplier: from 'supply_capacity.csv', column 'supply_capacity' for each 'Supplier'.
    -   Demand of each customer: from 'customer_demand.csv', column 'demand' for each 'Customers'.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize $\sum_{i}\sum_{j} \text{cost}_{i,j} \cdot x_{i,j}$, where $\text{cost}_{i,j}$ is the transportation cost per unit from supplier $i$ to customer $j$.
7.  **Formulate Constraints:**
    -   Supply Capacity Constraint: For each supplier $i$, the total quantity shipped from $i$ to all customers cannot exceed its supply capacity: $\sum_{j} x_{i,j} \leq \text{supply\_capacity}_i$.
    -   Demand Satisfaction Constraint: For each customer $j$, the total quantity received from all suppliers must exactly meet its demand: $\sum_{i} x_{i,j} = \text{demand}_j$.
    -   Non-negativity: For all $i, j$, $x_{i,j} \geq 0$.
[Abstract Model Plan END]
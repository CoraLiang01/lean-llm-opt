[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan for delivering goods from Walmart stores to customer groups, such that all customer demands are met, no store exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Stores (Supply nodes): S = {S1, S2, ..., S11} (from 'supply_capacity.csv' and 'transportation_costs.csv')
    - Customers (Demand nodes): C = {C1, C2, ..., C12} (from 'customer_demand.csv' and 'transportation_costs.csv')
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of goods transported from store s to customer c. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Transportation cost per unit from each store to each customer, from 'transportation_costs.csv' columns C1–C12 for each row S1–S11.
    -   Constraint coefficients:
        -   For supply constraints: Each store’s total outgoing shipments, using 'supply_capacity.csv' column 'supply_capacity' for each store.
        -   For demand constraints: Each customer’s total incoming shipments, using 'customer_demand.csv' column 'demand' for each customer.
    -   Constraint RHS (limits):
        -   Supply upper bounds: Each store’s 'supply_capacity'.
        -   Demand requirements: Each customer’s 'demand'.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all stores and customers of (transportation_cost[s, c] * x[s, c]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer c, the sum over all stores s of x[s, c] must equal the demand of customer c (from 'customer_demand.csv').
    -   Constraint 2 (Supply Capacity): For each store s, the sum over all customers c of x[s, c] must be less than or equal to the supply capacity of store s (from 'supply_capacity.csv').
    -   Constraint 3 (Non-negativity): For all s in S and c in C, x[s, c] ≥ 0.
[Abstract Model Plan END]
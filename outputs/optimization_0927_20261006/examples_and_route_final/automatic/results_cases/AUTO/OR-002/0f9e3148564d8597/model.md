[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan for delivering goods from Walmart stores to customer groups, such that all customer demands are met, no store exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Stores (S): All rows from 'supply_capacity.csv' (store IDs, e.g., S1, S2, ..., S11).
    - Customers (C): All rows from 'customer_demand.csv' (customer IDs, e.g., C1, C2, ..., C12).
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of goods transported from store s ∈ S to customer c ∈ C. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit (`cost[s, c]`): from 'transportation_costs.csv', columns C1–C12 for each store row.
    -   Store supply capacity (`supply_capacity[s]`): from 'supply_capacity.csv', column 'supply_capacity' for each store.
    -   Customer demand (`demand[c]`): from 'customer_demand.csv', column 'demand' for each customer.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all stores and customers of (cost[s, c] * x[s, c]).
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each customer c ∈ C, the sum over all stores s ∈ S of x[s, c] must equal demand[c] (i.e., all customer demands must be fully met).
    -   Supply Capacity: For each store s ∈ S, the sum over all customers c ∈ C of x[s, c] must be less than or equal to supply_capacity[s] (i.e., no store ships more than its available supply).
    -   Non-negativity: For all s ∈ S and c ∈ C, x[s, c] ≥ 0.
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan for delivering goods from Walmart stores to customer groups, such that all customer demands are met, no store exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Stores (sources): S1, S2, ..., S11 (from `supply_capacity.csv` and `transportation_costs.csv` rows)
    - Customers (destinations): C1, C2, ..., C12 (from `customer_demand.csv` and `transportation_costs.csv` columns)
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of goods transported from store `s` to customer group `c`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit (`cost[s, c]`): from `transportation_costs.csv`, columns C1–C12 for each store S1–S11.
    -   Store supply capacity (`supply_capacity[s]`): from `supply_capacity.csv`, column 'supply_capacity' for each store.
    -   Customer demand (`demand[c]`): from `customer_demand.csv`, column 'demand' for each customer.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all stores and customers of `cost[s, c] * x[s, c]`.
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer group `c`, the sum of goods received from all stores must equal that customer's demand:  
        `sum over s of x[s, c] = demand[c]` for all customers `c`.
    -   Constraint 2 (Supply Capacity): For each store `s`, the total goods shipped from that store to all customers must not exceed its supply capacity:  
        `sum over c of x[s, c] <= supply_capacity[s]` for all stores `s`.
    -   Constraint 3 (Non-negativity): All shipment quantities must be non-negative:  
        `x[s, c] >= 0` for all stores `s` and customers `c`.
[Abstract Model Plan END]
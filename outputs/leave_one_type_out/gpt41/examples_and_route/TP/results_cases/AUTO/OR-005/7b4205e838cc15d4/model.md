[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily shipment quantities from each distribution center (supplier) to each customer group, such that all customer demands are fully met, no distribution center exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Suppliers (distribution centers): as listed in `supply_capacity.csv` (supplier1, ..., supplier8)
    - Customers (customer groups): as listed in `customer_demand.csv` (demand1, ..., demand8)
4.  **Define Decision Variables:**
    -   `x[supplier, customer]` = quantity of goods shipped from supplier to customer. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit: from `transportation_costs.csv`, columns demand1–demand8, rows supply1–supply8.
    -   Supply capacity per supplier: from `supply_capacity.csv`, column 'supply_capacity'.
    -   Demand per customer: from `customer_demand.csv`, column 'demand'.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all suppliers and customers of (transportation cost per unit from supplier to customer) × (quantity shipped from supplier to customer).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer, the sum of shipments received from all suppliers must equal that customer's demand (i.e., for each customer j: sum over i of x[i, j] = demand[j]).
    -   Constraint 2 (Supply Capacity): For each supplier, the total quantity shipped out to all customers must not exceed that supplier's supply capacity (i.e., for each supplier i: sum over j of x[i, j] ≤ supply_capacity[i]).
    -   Constraint 3 (Non-negativity): All shipment quantities x[i, j] ≥ 0.
[Abstract Model Plan END]
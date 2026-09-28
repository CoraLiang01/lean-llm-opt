[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily shipment quantities from each distribution center (supplier) to each customer group, such that all customer demands are fully met, no distribution center exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Suppliers (distribution centers): as listed in `supply_capacity.csv` (e.g., supplier1, supplier2, ..., supplier8)
    - Customers (customer groups): as listed in `customer_demand.csv` (e.g., demand1, demand2, ..., demand8)
4.  **Define Decision Variables:**
    -   `x[supplier, customer]` = quantity of goods shipped from a given supplier to a given customer group per day. Type: GRB.CONTINUOUS (non-negative real numbers; can be fractional if not otherwise restricted).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit (`c[supplier, customer]`): from `transportation_costs.csv`, columns `demand1` to `demand8` for each row (supplier).
    -   Supply capacity per supplier (`supply_capacity[supplier]`): from `supply_capacity.csv`, column `supply_capacity`.
    -   Demand per customer group (`demand[customer]`): from `customer_demand.csv`, column `demand`.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all suppliers and customers of (transportation cost per unit) × (quantity shipped):  
    Minimize: sum over all suppliers and customers of `c[supplier, customer] * x[supplier, customer]`.
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer group, the total quantity received from all suppliers must equal its demand.  
        For each customer: sum over all suppliers of `x[supplier, customer]` = `demand[customer]`.
    -   Constraint 2 (Supply Capacity): For each supplier, the total quantity shipped to all customers must not exceed its supply capacity.  
        For each supplier: sum over all customers of `x[supplier, customer]` ≤ `supply_capacity[supplier]`.
    -   Constraint 3 (Non-negativity): All shipment quantities must be non-negative.  
        For all supplier-customer pairs: `x[supplier, customer]` ≥ 0.
[Abstract Model Plan END]
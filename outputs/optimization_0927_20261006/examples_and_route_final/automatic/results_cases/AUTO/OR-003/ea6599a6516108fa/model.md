[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal allocation of goods from multiple suppliers to multiple customer groups, ensuring all customer demands are met, no supplier exceeds its daily capacity, and total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are Suppliers (from 'supply_capacity.csv', e.g., S1–S10) and Customers (from 'customer_demand.csv', e.g., C1–C10).
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of goods transported from supplier `s` to customer `c`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Transportation cost per unit from each supplier to each customer, from 'transportation_costs.csv' columns C1–C10 for each supplier row.
    -   Supply capacity: Maximum daily supply per supplier, from 'supply_capacity.csv' column 'supply_capacity'.
    -   Demand: Daily demand per customer group, from 'customer_demand.csv' column 'demand'.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all suppliers and customers of (transportation cost per unit from supplier to customer) × (quantity shipped from that supplier to that customer).
7.  **Formulate Constraints:**
    -   Supply Capacity Constraint: For each supplier, the total quantity shipped to all customers does not exceed that supplier’s daily supply capacity.
    -   Demand Satisfaction Constraint: For each customer group, the total quantity received from all suppliers must exactly meet that customer’s daily demand.
    -   Non-negativity Constraint: All shipment quantities `x[s, c]` must be greater than or equal to zero.
[Abstract Model Plan END]
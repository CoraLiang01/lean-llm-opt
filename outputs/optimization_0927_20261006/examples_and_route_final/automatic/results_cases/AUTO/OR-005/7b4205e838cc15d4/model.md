[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily shipment quantities from each distribution center (supplier) to each customer group, such that all customer demands are fully met, no distribution center exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Suppliers (distribution centers), indexed by \( s \) (from 'Supplier' in supply_capacity.csv and 'Unnamed: 0' in transportation_costs.csv)
    - Customers (customer groups), indexed by \( c \) (from 'Customers' in customer_demand.csv and column headers in transportation_costs.csv)
4.  **Define Decision Variables:**
    -   \( x_{s,c} \) = quantity of goods shipped from supplier \( s \) to customer \( c \). Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit: from 'transportation_costs.csv', entry at row \( s \), column \( c \).
    -   Supply capacity per supplier: from 'supply_capacity.csv', 'supply_capacity' column for each 'Supplier' \( s \).
    -   Demand per customer: from 'customer_demand.csv', 'demand' column for each 'Customers' \( c \).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all suppliers and customers of (transportation cost per unit from \( s \) to \( c \)) times \( x_{s,c} \).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer \( c \), the sum of shipments received from all suppliers must equal the demand of \( c \) (i.e., \( \sum_{s} x_{s,c} = \) demand of \( c \)).
    -   Constraint 2 (Supply Capacity): For each supplier \( s \), the total quantity shipped out to all customers must not exceed the supply capacity of \( s \) (i.e., \( \sum_{c} x_{s,c} \leq \) supply_capacity of \( s \)).
    -   Constraint 3 (Non-negativity): All shipment quantities \( x_{s,c} \geq 0 \).
[Abstract Model Plan END]
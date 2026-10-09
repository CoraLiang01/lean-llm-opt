[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal shipment quantities from each production plant to each retail outlet, such that all customer demands are satisfied, no plant exceeds its production capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Plants (sources), indexed by \( s \) (from supply_capacity.csv and transportation_costs.csv, e.g., S1, S2, S3, S4)
    - Customers (destinations), indexed by \( c \) (from customer_demand.csv and transportation_costs.csv, e.g., C1, C2, C3, C4)
4.  **Define Decision Variables:**
    -   \( x_{s,c} \) = quantity of beverages shipped from plant \( s \) to customer \( c \). Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from plant \( s \) to customer \( c \): from 'transportation_costs.csv', columns ['C1', 'C2', 'C3', 'C4'] for each row (plant).
    -   Plant production capacity: from 'supply_capacity.csv', column 'supply_capacity' for each plant.
    -   Customer demand: from 'customer_demand.csv', column 'demand' for each customer.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all plants and customers of (transportation cost per unit from \( s \) to \( c \)) times (\( x_{s,c} \)).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer \( c \), the sum of shipments received from all plants must equal the customer's demand (i.e., \( \sum_{s} x_{s,c} = \) demand of \( c \)).
    -   Constraint 2 (Supply Capacity): For each plant \( s \), the total quantity shipped from that plant to all customers must not exceed its supply capacity (i.e., \( \sum_{c} x_{s,c} \leq \) supply_capacity of \( s \)).
    -   Constraint 3 (Non-negativity): All shipment quantities \( x_{s,c} \geq 0 \).
[Abstract Model Plan END]
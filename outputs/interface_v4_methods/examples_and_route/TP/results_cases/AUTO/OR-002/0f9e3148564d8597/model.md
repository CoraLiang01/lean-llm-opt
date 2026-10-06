[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal transportation plan for delivering goods from Walmart stores (suppliers) to customer groups, such that all customer demands are satisfied, no store exceeds its supply capacity, and the total transportation cost is minimized. All relevant data (demands, supply capacities, and per-unit transportation costs) are provided in three CSV files.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Stores (Suppliers): S = {S1, S2, ..., S11} (from 'supply_capacity.csv' and 'transportation_costs.csv')
    - Customers: C = {C1, C2, ..., C12} (from 'customer_demand.csv' and 'transportation_costs.csv')
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of goods transported from store s to customer c. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Per-unit transportation cost from store s to customer c, from 'transportation_costs.csv' (columns C1–C12, rows S1–S11).
    -   Constraint coefficients:
        -   For supply constraints: Each store's total outgoing shipments, summed over all customers, from 'supply_capacity.csv' (column 'supply_capacity').
        -   For demand constraints: Each customer's total incoming shipments, summed over all stores, from 'customer_demand.csv' (column 'demand').
    -   Constraint RHS (limits):
        -   Supply upper bounds: Each store's 'supply_capacity'.
        -   Demand requirements: Each customer's 'demand'.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all stores and customers of (transportation cost per unit from s to c) × (quantity shipped from s to c):  
    "Minimize sum_{s in S, c in C} transportation_costs[s, c] * x[s, c]"
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer c, the sum of shipments received from all stores must equal that customer's demand:  
        "For all c in C: sum_{s in S} x[s, c] = customer_demand[c]"
    -   Constraint 2 (Supply Capacity): For each store s, the sum of shipments sent to all customers must not exceed that store's supply capacity:  
        "For all s in S: sum_{c in C} x[s, c] <= supply_capacity[s]"
    -   Constraint 3 (Non-negativity): All shipment quantities must be non-negative:  
        "For all s in S, c in C: x[s, c] >= 0"
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily shipment quantities from each warehouse to each retail store, such that all store demands are satisfied, no warehouse exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Warehouses (from supply_capacity.csv and transportation_costs.csv, labeled S1–S10)
    - Customers/Stores (from customer_demand.csv and transportation_costs.csv, labeled C1–C10)
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of product shipped from warehouse `s` to customer/store `c`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Transportation cost per unit from warehouse to store, from transportation_costs.csv columns C1–C10 for each row S1–S10.
    -   Constraint coefficients:
        -   Demand for each customer/store: customer_demand.csv, column 'demand' for each 'customer'.
        -   Supply capacity for each warehouse: supply_capacity.csv, column 'supply_capacity' for each warehouse.
    -   Constraint RHS (limits):
        -   For demand constraints: customer_demand.csv, 'demand'.
        -   For supply constraints: supply_capacity.csv, 'supply_capacity'.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all warehouses and stores of (transportation_costs[s, c] * x[s, c]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer/store c, the sum of shipments received from all warehouses must equal the demand, i.e., sum over s of x[s, c] = demand[c].
    -   Constraint 2 (Supply Capacity): For each warehouse s, the sum of shipments sent to all customers/stores must not exceed the warehouse's supply capacity, i.e., sum over c of x[s, c] ≤ supply_capacity[s].
    -   Constraint 3 (Non-negativity): All shipment variables x[s, c] ≥ 0.
[Abstract Model Plan END]
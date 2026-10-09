[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal quantities of products to ship from each warehouse to each retail store, such that all store demands are satisfied, no warehouse exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are Warehouses (from `supply_capacity.csv` and `transportation_costs.csv` rows, e.g., S1–S10) and Customers/Stores (from `customer_demand.csv` and `transportation_costs.csv` columns, e.g., C1–C10).
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of product shipped from warehouse `s` to customer/store `c`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Transportation cost per unit from warehouse `s` to customer `c`, from `transportation_costs.csv` (field: cost at [row s, column c]).
    -   Constraint coefficients:
        -   Demand for each customer/store, from `customer_demand.csv` (field: 'demand' for each 'customer').
        -   Supply capacity for each warehouse, from `supply_capacity.csv` (field: 'supply_capacity' for each warehouse).
    -   Constraint RHS (limits): Customer demands and warehouse supply capacities as above.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all warehouses and customers of (transportation cost per unit from warehouse to customer) × (quantity shipped from warehouse to customer).
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer/store, the total quantity received from all warehouses must be at least (or exactly) equal to its demand (sum over warehouses of `x[s, c]` = demand for each customer `c`).
    -   Constraint 2 (Supply Capacity): For each warehouse, the total quantity shipped to all customers must not exceed its supply capacity (sum over customers of `x[s, c]` ≤ supply capacity for each warehouse `s`).
    -   Constraint 3 (Non-negativity): All shipment quantities must be non-negative (`x[s, c]` ≥ 0 for all warehouses `s` and customers `c`).
[Abstract Model Plan END]
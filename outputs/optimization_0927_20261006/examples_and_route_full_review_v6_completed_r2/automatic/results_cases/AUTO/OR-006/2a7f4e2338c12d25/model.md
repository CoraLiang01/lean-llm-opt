[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily shipment quantities from each warehouse to each retail store, such that all store demands are satisfied, no warehouse exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are Warehouses (from `supply_capacity.csv` and `transportation_costs.csv` rows, e.g., S1–S10) and Customers/Stores (from `customer_demand.csv` and `transportation_costs.csv` columns, e.g., C1–C10).
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of product shipped from warehouse `s` to customer/store `c`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Transportation cost per unit from warehouse `s` to customer `c`, from `transportation_costs.csv` (field: cost at [row s, column c]).
    -   Constraint coefficients:
        -   For demand satisfaction: Each `x[s, c]` contributes to meeting customer `c`'s demand (from `customer_demand.csv`, field: `demand`).
        -   For supply capacity: Each `x[s, c]` consumes supply from warehouse `s` (from `supply_capacity.csv`, field: `supply_capacity`).
    -   Constraint RHS (limits):
        -   Customer demand: `demand` for each customer from `customer_demand.csv`.
        -   Warehouse supply: `supply_capacity` for each warehouse from `supply_capacity.csv`.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., sum over all warehouses and customers of (transportation cost per unit from `s` to `c`) × `x[s, c]`.
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer/store `c`, the sum over all warehouses `s` of `x[s, c]` must be at least equal to the demand of customer `c` (i.e., sum_s x[s, c] = demand[c]).
    -   Constraint 2 (Supply Capacity): For each warehouse `s`, the sum over all customers `c` of `x[s, c]` must not exceed the supply capacity of warehouse `s` (i.e., sum_c x[s, c] ≤ supply_capacity[s]).
    -   Constraint 3 (Non-negativity): All shipment quantities `x[s, c]` must be greater than or equal to zero.
[Abstract Model Plan END]
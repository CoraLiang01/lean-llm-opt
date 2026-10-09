[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily shipment quantities from each warehouse to each GreenMart store, such that all store demands are satisfied, no warehouse exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Warehouses (from `supply_capacity.csv`, column 'region', e.g., S1, S2, S3, S4, S5)
    - Stores (from `customer_demand.csv`, column 'customer', e.g., D1, D2, D3, D4, D5)
4.  **Define Decision Variables:**
    -   `x[w, d]` = Quantity of product shipped from warehouse `w` to store `d` per day. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from warehouse to store: from `transportation_costs.csv`, columns D1–D5, indexed by warehouse (row 'Unnamed: 0').
    -   Store daily demand: from `customer_demand.csv`, column 'demand', indexed by 'customer'.
    -   Warehouse daily supply capacity: from `supply_capacity.csv`, column 'supply_capacity', indexed by 'region'.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all warehouses and stores of (transportation cost per unit from warehouse to store) × (quantity shipped from warehouse to store):  
    Minimize ∑<sub>w∈Warehouses</sub> ∑<sub>d∈Stores</sub> [transportation_costs[w, d] × x[w, d]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each store, the total quantity received from all warehouses must meet its daily demand:  
        For all d ∈ Stores: ∑<sub>w∈Warehouses</sub> x[w, d] = customer_demand[d]
    -   Constraint 2 (Supply Capacity): For each warehouse, the total quantity shipped to all stores must not exceed its daily supply capacity:  
        For all w ∈ Warehouses: ∑<sub>d∈Stores</sub> x[w, d] ≤ supply_capacity[w]
    -   Constraint 3 (Non-negativity): For all w ∈ Warehouses, d ∈ Stores: x[w, d] ≥ 0
[Abstract Model Plan END]
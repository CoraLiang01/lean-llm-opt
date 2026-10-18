[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select an optimal subset of potential warehouse locations to open, and assign all customer demands to these warehouses, so as to minimize the total cost (sum of fixed warehouse opening costs and variable transportation costs), subject to warehouse capacity limits and full demand satisfaction.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a capacitated facility location (fixed-charge) model.
3.  **Define Index Sets:** The primary indices are:
    - Warehouses (set W): from 'Warehouse ID' in warehouse.csv and cost.csv.
    - Customers (set C): from 'Customer ID' in demand.csv and cost.csv.
4.  **Define Decision Variables:**
    - `y[w]` = 1 if warehouse w is opened, 0 otherwise. Type: GRB.BINARY.
    - `x[w,c]` = amount of customer c's demand served from warehouse w. Type: GRB.CONTINUOUS (≥ 0).
5.  **Identify Parameters (from Schema):**
    - Fixed opening cost for each warehouse: 'Fixed_Cost' from warehouse.csv.
    - Warehouse capacity: 'Capacity' from warehouse.csv.
    - Customer demand: 'Demand' from demand.csv.
    - Transportation cost per unit from warehouse w to customer c: cost.csv, columns 'C1'...'C20' indexed by warehouse and customer.
6.  **Formulate Objective:** Minimize the total cost, which is the sum over all warehouses of (fixed opening cost × y[w]) plus the sum over all warehouse-customer pairs of (transportation cost per unit × x[w,c]).
7.  **Formulate Constraints:**
    - Demand Satisfaction: For each customer c, the sum over all warehouses w of x[w,c] equals the total demand of customer c.
    - Warehouse Capacity: For each warehouse w, the sum over all customers c of x[w,c] is less than or equal to the capacity of warehouse w multiplied by y[w] (i.e., only open warehouses can serve demand, and not beyond their capacity).
    - Non-negativity: All x[w,c] ≥ 0.
    - Binary: All y[w] ∈ {0,1}.
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal shipment plan from multiple suppliers to multiple customers, minimizing the total cost, which includes both per-unit transportation costs and fixed route activation costs. The plan must satisfy all customer demands, not exceed supplier capacities, and only incur a fixed cost if a route is used.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) Fixed-Charge Transportation problem.
3.  **Define Index Sets:** The primary indices are Suppliers (from 'Supplier' in supplier_capacity.csv) and Customers (from 'Customer' in customer_demand.csv).
4.  **Define Decision Variables:**
    -   `x[i,j]` = Quantity shipped from supplier i to customer j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i,j]` = 1 if any positive amount is shipped from supplier i to customer j (i.e., route i-j is used), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable transportation cost per unit (`c[i,j]`): from 'route_variable_costs.csv', columns 'C1'...'C12' for each supplier-customer pair.
    -   Fixed route activation cost (`f[i,j]`): from 'route_fixed_costs.csv', columns 'C1'...'C12' for each supplier-customer pair.
    -   Supplier capacity (`SupplyCapacity[i]`): from 'supplier_capacity.csv', column 'SupplyCapacity' for each supplier.
    -   Customer demand (`Demand[j]`): from 'customer_demand.csv', column 'Demand' for each customer.
6.  **Formulate Objective:** Minimize the total cost, which is the sum over all supplier-customer pairs of (variable transportation cost per unit * quantity shipped) plus (fixed route activation cost * binary route-use variable):  
    Minimize sum over i,j of [c[i,j] * x[i,j] + f[i,j] * y[i,j]].
7.  **Formulate Constraints:**
    -   Constraint 1 (Customer Demand Satisfaction): For each customer j, the total quantity received from all suppliers must equal the customer's demand:  
        sum over i of x[i,j] = Demand[j] for all j.
    -   Constraint 2 (Supplier Capacity): For each supplier i, the total quantity shipped to all customers must not exceed the supplier's capacity:  
        sum over j of x[i,j] ≤ SupplyCapacity[i] for all i.
    -   Constraint 3 (Route Activation Linking): For each supplier-customer pair (i,j), the quantity shipped must be zero unless the route is activated; i.e.,  
        x[i,j] ≤ M[i,j] * y[i,j], where M[i,j] is a sufficiently large upper bound (e.g., min{SupplyCapacity[i], Demand[j]}).
    -   Constraint 4 (Nonnegativity): x[i,j] ≥ 0 for all i,j.
    -   Constraint 5 (Binary Route Use): y[i,j] ∈ {0,1} for all i,j.
[Abstract Model Plan END]
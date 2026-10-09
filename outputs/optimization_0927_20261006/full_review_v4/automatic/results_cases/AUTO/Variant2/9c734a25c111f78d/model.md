[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal shipment plan from multiple suppliers to multiple customers, minimizing the total cost, which includes both per-unit transportation costs and fixed route activation costs. The plan must satisfy all customer demands, not exceed supplier capacities, and only incur a fixed cost for a route if it is used.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem, specifically a Fixed-Charge Transportation Problem.
3.  **Define Index Sets:** The primary indices are:
    - Suppliers (set S), from the 'Supplier' column in supplier_capacity.csv.
    - Customers (set C), from the 'Customer' column in customer_demand.csv.
4.  **Define Decision Variables:**
    -   `x[i,j]` = quantity shipped from supplier i to customer j. Type: GRB.CONTINUOUS (nonnegative).
    -   `y[i,j]` = 1 if any positive amount is shipped from supplier i to customer j (i.e., route i-j is used), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Variable transportation cost per unit (`c[i,j]`): from route_variable_costs.csv, columns 'C1'...'C12' for each supplier.
    -   Fixed route activation cost (`f[i,j]`): from route_fixed_costs.csv, columns 'C1'...'C12' for each supplier.
    -   Supplier capacity (`SupplyCapacity[i]`): from supplier_capacity.csv, column 'SupplyCapacity'.
    -   Customer demand (`Demand[j]`): from customer_demand.csv, column 'Demand'.
6.  **Formulate Objective:** Minimize the total cost, which is the sum over all supplier-customer pairs of (variable transportation cost per unit * quantity shipped) plus (fixed route activation cost * route-use binary variable):
        Minimize sum_{i in S, j in C} [ c[i,j] * x[i,j] + f[i,j] * y[i,j] ]
7.  **Formulate Constraints:**
    -   Constraint 1 (Customer Demand Satisfaction): For each customer j, the total quantity received from all suppliers must equal the customer's demand:
            sum_{i in S} x[i,j] = Demand[j]   for all j in C
    -   Constraint 2 (Supplier Capacity Limit): For each supplier i, the total quantity shipped to all customers must not exceed the supplier's capacity:
            sum_{j in C} x[i,j] <= SupplyCapacity[i]   for all i in S
    -   Constraint 3 (Route Activation Linking): For each supplier-customer pair (i,j), shipments on a route are only allowed if the route is activated:
            x[i,j] <= M[i,j] * y[i,j]   for all i in S, j in C
        where M[i,j] is a sufficiently large upper bound (e.g., min{SupplyCapacity[i], Demand[j]}).
    -   Constraint 4 (Nonnegativity): x[i,j] >= 0   for all i in S, j in C
    -   Constraint 5 (Binary Route Use): y[i,j] in {0,1}   for all i in S, j in C
[Abstract Model Plan END]
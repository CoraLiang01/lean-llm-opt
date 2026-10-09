[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal allocation of goods from multiple suppliers to multiple customer groups, such that all customer demands are satisfied, no supplier exceeds its daily supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Suppliers (from `supply_capacity.csv`, column `'Unnamed: 0'`)
    - Customers (from `customer_demand.csv`, column `'customer'`)
4.  **Define Decision Variables:**
    -   `x[s, c]` = quantity of goods transported from supplier `s` to customer `c`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from supplier to customer: from `transportation_costs.csv`, columns `'C1'` to `'C10'` (indexed by supplier and customer).
    -   Supplier daily supply capacity: from `supply_capacity.csv`, column `'supply_capacity'` (indexed by supplier).
    -   Customer daily demand: from `customer_demand.csv`, column `'demand'` (indexed by customer).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all suppliers and customers of (transportation cost per unit from supplier to customer) × (quantity shipped from supplier to customer):  
        Minimize ∑ₛ ∑𝚌 [transportation_costs[s, c] * x[s, c]]
7.  **Formulate Constraints:**
    -   Supplier Capacity Constraints: For each supplier, the total quantity shipped to all customers cannot exceed its supply capacity:  
        For all suppliers s: ∑𝚌 x[s, c] ≤ supply_capacity[s]
    -   Customer Demand Constraints: For each customer, the total quantity received from all suppliers must exactly meet its demand:  
        For all customers c: ∑ₛ x[s, c] = demand[c]
    -   Non-negativity Constraints: For all suppliers s and customers c:  
        x[s, c] ≥ 0
[Abstract Model Plan END]
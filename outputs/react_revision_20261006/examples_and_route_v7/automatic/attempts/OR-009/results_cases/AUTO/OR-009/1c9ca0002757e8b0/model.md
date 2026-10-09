[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal shipment quantities of beverages from each production facility (plant) to each retail outlet (customer), such that all customer demands are satisfied, no plant exceeds its production capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Plants (Sources): S = {S1, S2, S3, S4} (from 'supply_capacity.csv' and 'transportation_costs.csv')
    - Customers (Destinations): C = {C1, C2, C3, C4} (from 'customer_demand.csv' and 'transportation_costs.csv')
4.  **Define Decision Variables:**
    -   `x[s, c]` = Quantity of beverages shipped from plant s ∈ S to customer c ∈ C. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from plant s to customer c: from 'transportation_costs.csv', columns ['C1', 'C2', 'C3', 'C4'] indexed by 'Unnamed: 0' (plant).
    -   Plant production capacity: from 'supply_capacity.csv', column 'supply_capacity' indexed by 'Unnamed: 0' (plant).
    -   Customer demand: from 'customer_demand.csv', column 'demand' indexed by 'customer'.
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all plants and customers of (transportation cost per unit from s to c) × (quantity shipped from s to c):  
    Minimize ∑ₛ∈S ∑_c∈C [transportation_costs[s, c] * x[s, c]]
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each customer c ∈ C, the total quantity received from all plants must equal that customer's demand:  
        ∑ₛ∈S x[s, c] = customer_demand[c]  ∀ c ∈ C
    -   Constraint 2 (Supply Capacity): For each plant s ∈ S, the total quantity shipped from that plant to all customers must not exceed its supply capacity:  
        ∑_c∈C x[s, c] ≤ supply_capacity[s]  ∀ s ∈ S
    -   Constraint 3 (Non-negativity): All shipment quantities must be non-negative:  
        x[s, c] ≥ 0  ∀ s ∈ S, c ∈ C
[Abstract Model Plan END]
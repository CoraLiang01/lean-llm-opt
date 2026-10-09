[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal shipment quantities from each production plant to each retail outlet, such that all customer demands are satisfied, no plant exceeds its production capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Plants (sources), indexed by \( s \) (from 'supply_capacity.csv', column 'Unnamed: 0')
    - Customers (destinations), indexed by \( c \) (from 'customer_demand.csv', column 'customer')
4.  **Define Decision Variables:**
    -   \( x_{s,c} \) = quantity of beverages shipped from plant \( s \) to customer \( c \). Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit from plant \( s \) to customer \( c \): from 'transportation_costs.csv', columns 'C1', 'C2', 'C3', 'C4' (indexed by plant in 'Unnamed: 0').
    -   Plant production capacity: from 'supply_capacity.csv', column 'supply_capacity' (indexed by plant in 'Unnamed: 0').
    -   Customer demand: from 'customer_demand.csv', column 'demand' (indexed by customer in 'customer').
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize the sum over all plants and customers of (transportation cost per unit from plant \( s \) to customer \( c \)) times (\( x_{s,c} \)).
7.  **Formulate Constraints:**
    -   Supply constraint (for each plant): The total quantity shipped from each plant to all customers cannot exceed that plant’s supply capacity.
    -   Demand constraint (for each customer): The total quantity received by each customer from all plants must exactly meet that customer’s demand.
    -   Non-negativity: All shipment quantities \( x_{s,c} \) must be greater than or equal to zero.
[Abstract Model Plan END]
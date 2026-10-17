[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily transportation plan from multiple Amazon distribution centers to various customer groups, such that all customer demands are satisfied, no distribution center exceeds its supply capacity, and the total transportation cost is minimized.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) transportation problem.
3.  **Define Index Sets:** The primary indices are:
    - Distribution Centers (Sources): indexed by \( s \), from the 'Unnamed: 0' column in supply_capacity.csv and transportation_costs.csv (e.g., S1, S2, ..., S18).
    - Customer Groups (Destinations): indexed by \( c \), from the 'customer' column in customer_demand.csv and the columns C1–C18 in transportation_costs.csv.
4.  **Define Decision Variables:**
    -   \( x_{s,c} \) = Quantity of goods transported from distribution center \( s \) to customer group \( c \). Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Transportation cost per unit (\( cost_{s,c} \)): from transportation_costs.csv, columns C1–C18 for each source \( s \).
    -   Supply capacity (\( supply\_capacity_s \)): from supply_capacity.csv, 'supply_capacity' for each source \( s \).
    -   Customer demand (\( demand_c \)): from customer_demand.csv, 'demand' for each customer group \( c \).
6.  **Formulate Objective:** Minimize the total transportation cost, i.e., minimize \( \sum_{s} \sum_{c} cost_{s,c} \cdot x_{s,c} \).
7.  **Formulate Constraints:**
    -   Supply Capacity Constraints: For each distribution center \( s \), \( \sum_{c} x_{s,c} \leq supply\_capacity_s \) (do not exceed supply at any center).
    -   Demand Satisfaction Constraints: For each customer group \( c \), \( \sum_{s} x_{s,c} = demand_c \) (fully meet each customer's demand).
    -   Non-negativity: For all \( s, c \), \( x_{s,c} \geq 0 \).
[Abstract Model Plan END]
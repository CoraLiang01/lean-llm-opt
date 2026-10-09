[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material costs), while satisfying blending proportion requirements for each brand, raw material supply limits, and a minimum production requirement for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Blending Linear Programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Raw Grades: {I, II, III}
    - Brands: {Red, Yellow, Blue}
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS (non-negative real variables).
5.  **Identify Parameters (from Schema):**
    -   From 30-1.csv (all rows required):
        - `supply[g]`: Daily supply limit for grade `g` ("Daily Supply (kg)")
        - `cost[g]`: Unit cost for grade `g` ("Cost (CNY/kg)")
    -   From 30-2.csv (all rows required):
        - `price[b]`: Selling price for brand `b` ("Selling Price (CNY/kg)")
        - `blend_req[b]`: Blending requirements for brand `b` ("Blending Requirements") — parsed into lower/upper bounds on proportions of certain grades in each brand.
6.  **Formulate Objective:** Maximize total net profit, calculated as:
    - Total sales revenue: sum over all brands of (total kg produced of brand `b`) × (selling price of brand `b`)
    - Minus total raw material cost: sum over all grades and brands of (kg of grade `g` used in brand `b`) × (cost of grade `g`)
    - In formula: Maximize  
      sum_b [ price[b] × sum_g x[g, b] ] − sum_g sum_b [ cost[g] × x[g, b] ]
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand, enforce upper and/or lower bounds on the proportion of specified grades in the blend:
        - For each brand `b` and each grade `g` with a requirement,  
          (sum_g x[g, b]) > 0 ⇒  
          lower_bound[g, b] ≤ x[g, b] / sum_g x[g, b] ≤ upper_bound[g, b]
        - For example:
            - Red: Proportion of I < 10%; Proportion of II > 50%
            - Yellow: Proportion of III < 70%; Proportion of I > 20%
            - Blue: Proportion of I < 50%; Proportion of II > 10%
    -   Constraint 2 (Raw Material Supply): For each grade `g`,  
        sum_b x[g, b] ≤ supply[g]
    -   Constraint 3 (Minimum Production for Red):  
        sum_g x[g, 'Red'] ≥ 2000
    -   Constraint 4 (Non-negativity):  
        x[g, b] ≥ 0 for all grades and brands
[Abstract Model Plan END]
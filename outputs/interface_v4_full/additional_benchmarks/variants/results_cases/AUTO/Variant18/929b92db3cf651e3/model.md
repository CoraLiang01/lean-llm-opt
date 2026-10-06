[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum number of stock rolls required to meet or exceed the demand for each item, using only a fixed set of feasible cutting patterns. Each pattern can be used any nonnegative integer number of times, and overproduction is allowed.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a cutting-stock (set covering) model with integer variables.
3.  **Define Index Sets:** The primary indices are:
    - Items (from item_demand.csv): {A, B, C, D, E}
    - Cutting Patterns (from cutting_patterns.csv): {P1, P2, ..., P10}
4.  **Define Decision Variables:**
    -   `y_p` = Number of times cutting pattern p is used (i.e., number of rolls cut with pattern p). Type: GRB.INTEGER (nonnegative integer).
5.  **Identify Parameters (from Schema):**
    -   For each item i and pattern p, the number of pieces of item i produced by pattern p: from columns 'A', 'B', 'C', 'D', 'E' in cutting_patterns.csv.
    -   For each item i, the demand: from column 'Demand' in item_demand.csv.
6.  **Formulate Objective:** Minimize the total number of rolls used, i.e., minimize the sum over all patterns of y_p.
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each item i, the total number of pieces produced across all patterns must be at least the demand for that item. That is, for each item i: sum over all patterns p of (number of pieces of item i in pattern p) * y_p >= demand for item i.
    -   Constraint 2 (Nonnegativity and Integrality): For each pattern p, y_p >= 0 and integer.
[Abstract Model Plan END]
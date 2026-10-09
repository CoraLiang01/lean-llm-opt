[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum number of stock rolls to cut in order to meet or exceed the demand for each item, using only a fixed set of feasible cutting patterns. Overproduction is allowed, and the number of times each pattern is used must be a nonnegative integer.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a cutting-stock (set covering) model with integer variables.
3.  **Define Index Sets:** The primary indices are:
    - Items (from item_demand.csv): set I
    - Cutting patterns (from cutting_patterns.csv): set P
4.  **Define Decision Variables:**
    -   `y_p` = Number of times cutting pattern p ∈ P is used (i.e., number of rolls cut with pattern p). Type: GRB.INTEGER (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   Demand for each item i: from item_demand.csv, column 'Demand'.
    -   Number of pieces of item i produced by pattern p: from cutting_patterns.csv, columns 'A', 'B', 'C', 'D', 'E' (one column per item).
6.  **Formulate Objective:** Minimize the total number of rolls used, i.e., minimize the sum over all patterns p of y_p.
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each item i ∈ I, the total number of pieces produced across all patterns must be at least the demand for item i. That is, for each i, sum over p of (number of pieces of item i in pattern p) × y_p ≥ demand for item i.
    -   Constraint 2 (Nonnegativity and Integrality): For each pattern p ∈ P, y_p ≥ 0 and integer.
[Abstract Model Plan END]
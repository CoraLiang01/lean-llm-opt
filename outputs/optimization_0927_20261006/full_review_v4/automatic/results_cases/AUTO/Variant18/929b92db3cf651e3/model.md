[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum number of stock rolls required to meet or exceed the demand for each item, using only a fixed set of feasible cutting patterns. Each pattern can be used any nonnegative integer number of times, and overproduction is allowed.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a cutting-stock (set covering) model with integer variables.
3.  **Define Index Sets:** The primary indices are:
    - Items (from item_demand.csv): set I
    - Cutting patterns (from cutting_patterns.csv): set P
4.  **Define Decision Variables:**
    -   `y_p` = Number of times cutting pattern p ∈ P is used (i.e., number of rolls cut with pattern p). Type: GRB.INTEGER (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   For each item i ∈ I:
        -   Demand: `demand[i]` (from 'Demand' column in item_demand.csv)
    -   For each pattern p ∈ P and item i ∈ I:
        -   Pieces produced: `a[p,i]` (from columns 'A', 'B', 'C', 'D', 'E' in cutting_patterns.csv, mapping item i to its column)
6.  **Formulate Objective:** Minimize the total number of rolls used, i.e., minimize sum over all patterns p of y_p.
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each item i ∈ I, the total number of pieces produced across all patterns must be at least the demand for item i:
        -   sum over all patterns p of (a[p,i] * y_p) ≥ demand[i]
    -   Nonnegativity and Integrality: For each pattern p ∈ P, y_p ≥ 0 and integer.
[Abstract Model Plan END]
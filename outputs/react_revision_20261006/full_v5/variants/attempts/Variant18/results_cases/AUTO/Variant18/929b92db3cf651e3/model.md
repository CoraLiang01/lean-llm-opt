[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum number of stock rolls required to meet or exceed the demand for each item, using only a fixed set of feasible cutting patterns. Each pattern can be used any nonnegative integer number of times, and overproduction is allowed.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a cutting-stock (set covering) model with integer variables.
3.  **Define Index Sets:** The primary indices are:
    - Patterns (from the 'Pattern' column in cutting_patterns.csv)
    - Items (from the 'Item' column in item_demand.csv and the item columns in cutting_patterns.csv)
4.  **Define Decision Variables:**
    -   `y[p]` = Number of times cutting pattern p is used (i.e., number of rolls cut with pattern p). Type: GRB.INTEGER (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   For each pattern p and item i, the number of pieces of item i produced by pattern p: from columns 'A', 'B', 'C', 'D', 'E' in cutting_patterns.csv.
    -   For each item i, the demand to be satisfied: from 'Demand' column in item_demand.csv.
6.  **Formulate Objective:** Minimize the total number of rolls used, i.e., minimize the sum over all patterns p of y[p].
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each item i, the total number of pieces produced across all patterns must be at least the demand for item i. That is, for each item i: sum over all patterns p of (number of pieces of item i in pattern p) * y[p] ≥ demand for item i.
    -   Constraint 2 (Nonnegativity and Integrality): For each pattern p, y[p] ≥ 0 and integer.
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum number of stock rolls to cut, using a fixed set of feasible cutting patterns, such that the demand for each item is satisfied (overproduction allowed), with integer numbers of rolls per pattern.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) cutting-stock problem.
3.  **Define Index Sets:** The primary indices are:
    - Patterns (from 'Pattern' in cutting_patterns.csv)
    - Items (from 'Item' in item_demand.csv and columns 'A', 'B', 'C', 'D', 'E' in cutting_patterns.csv)
4.  **Define Decision Variables:**
    -   `y[p]` = Number of times pattern p is used (i.e., number of rolls cut with pattern p). Type: GRB.INTEGER (nonnegative).
5.  **Identify Parameters (from Schema):**
    -   For each item i and pattern p, the number of pieces of item i produced by pattern p: from columns 'A', 'B', 'C', 'D', 'E' in cutting_patterns.csv.
    -   For each item i, the demand: from 'Demand' in item_demand.csv.
6.  **Formulate Objective:** Minimize the total number of rolls used, i.e., minimize sum over all patterns p of y[p].
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each item i, the total number of pieces produced across all patterns must be at least the demand for item i. That is, for each item i: sum over all patterns p of (number of pieces of item i in pattern p) * y[p] ≥ demand for item i.
    -   Nonnegativity and Integrality: For each pattern p, y[p] ≥ 0 and integer.
[Abstract Model Plan END]
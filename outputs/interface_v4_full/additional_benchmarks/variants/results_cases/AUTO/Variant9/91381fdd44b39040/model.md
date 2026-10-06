[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum number of standard rolls to cut, by selecting how many times to use each feasible cutting pattern, such that the demand for each item type is satisfied or exceeded. Each pattern specifies how many units of each item type are produced per standard roll cut with that pattern.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically an integer cutting-stock pattern-selection model.
3.  **Define Index Sets:** The primary indices are:
    - Patterns (from 'Pattern' column in cutting_patterns.csv)
    - Item types (from 'Item' column in item_demand.csv and columns 'A', 'B', 'C', 'D' in cutting_patterns.csv)
4.  **Define Decision Variables:**
    -   `y[p]` = Number of standard rolls cut using pattern p (for each pattern p). Type: GRB.INTEGER (nonnegative integer).
5.  **Identify Parameters (from Schema):**
    -   Pattern composition: For each pattern p and item i, the number of units of item i produced by pattern p is from cutting_patterns.csv columns 'A', 'B', 'C', 'D'.
    -   Demand: For each item i, the required demand is from item_demand.csv column 'Demand'.
6.  **Formulate Objective:** Minimize the total number of standard rolls used, i.e., minimize the sum over all patterns p of y[p].
7.  **Formulate Constraints:**
    -   Constraint 1 (Demand Satisfaction): For each item type i (A, B, C, D), the total number of units produced across all patterns must be at least the demand for that item. That is, for each i: sum over all patterns p of (number of units of item i in pattern p) * y[p] ≥ demand for item i.
    -   Constraint 2 (Nonnegativity and Integrality): For each pattern p, y[p] ≥ 0 and integer.
[Abstract Model Plan END]
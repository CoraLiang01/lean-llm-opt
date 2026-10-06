[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum number of standard rolls to cut, by selecting how many times to use each feasible cutting pattern, such that the demand for each item type is satisfied or exceeded. Each pattern specifies how many units of each item type are produced per roll.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically an integer cutting-stock pattern-selection model.
3.  **Define Index Sets:** The primary indices are:
    - Patterns (from cutting_patterns.csv, column 'Pattern')
    - Item types (from item_demand.csv, column 'Item')
4.  **Define Decision Variables:**
    -   `y_p` = Number of standard rolls cut using pattern p (for each pattern p). Type: GRB.INTEGER (nonnegative integer).
5.  **Identify Parameters (from Schema):**
    -   Pattern composition: For each pattern p and item i, the number of units of item i produced by pattern p (from cutting_patterns.csv, columns 'A', 'B', 'C', 'D').
    -   Demand: For each item i, the required demand (from item_demand.csv, column 'Demand').
6.  **Formulate Objective:** Minimize the total number of standard rolls used, i.e., minimize the sum over all patterns of y_p.
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each item type i, the total number of units produced across all patterns must be at least the demand for that item. That is, for each item i: sum over all patterns p of (number of units of item i in pattern p) * y_p ≥ demand for item i.
    -   Nonnegativity and Integrality: For each pattern p, y_p ≥ 0 and integer.
[Abstract Model Plan END]
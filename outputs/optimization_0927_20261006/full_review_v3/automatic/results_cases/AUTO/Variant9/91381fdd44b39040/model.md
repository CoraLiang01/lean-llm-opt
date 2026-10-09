[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum number of standard rolls to cut, by selecting how many times to use each feasible cutting pattern, such that the demand for each item type is satisfied or exceeded. The model must use integer variables for the number of times each pattern is used, ensure nonnegativity, and cover all item demands.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically an integer cutting-stock pattern-selection model.
3.  **Define Index Sets:** The primary indices are:
    - Patterns (set of all feasible cutting patterns, from 'Pattern' in cutting_patterns.csv)
    - Items (set of all item types, from 'Item' in item_demand.csv)
4.  **Define Decision Variables:**
    - `y[p]` = Number of standard rolls cut using pattern p. Type: GRB.INTEGER (nonnegative).
5.  **Identify Parameters (from Schema):**
    - Pattern composition: Number of units of each item i produced by pattern p, from columns 'A', 'B', 'C', 'D' in cutting_patterns.csv.
    - Demand for each item i, from 'Demand' in item_demand.csv.
6.  **Formulate Objective:** Minimize the total number of standard rolls used, i.e., minimize the sum over all patterns p of y[p].
7.  **Formulate Constraints:**
    - Demand Satisfaction: For each item i, the sum over all patterns p of (number of units of item i produced by pattern p) times y[p] must be greater than or equal to the demand for item i.
    - Nonnegativity and Integrality: For each pattern p, y[p] ≥ 0 and integer.
[Abstract Model Plan END]
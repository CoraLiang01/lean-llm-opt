[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum number of standard rolls to cut, by selecting how many times to use each feasible cutting pattern, such that the demand for each item type is satisfied or exceeded. Each pattern specifies how many units of each item type are produced per roll.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically an integer cutting-stock pattern-selection model.
3.  **Define Index Sets:** The primary indices are:
    - Patterns (from 'Pattern' in cutting_patterns.csv)
    - Item types (from 'Item' in item_demand.csv and columns 'A', 'B', 'C', 'D' in cutting_patterns.csv)
4.  **Define Decision Variables:**
    - `y[p]` = Number of standard rolls cut using pattern p. Type: GRB.INTEGER (nonnegative integer).
5.  **Identify Parameters (from Schema):**
    - Pattern composition: For each pattern p and item i, the number of units of item i produced by pattern p comes from cutting_patterns.csv columns ['A', 'B', 'C', 'D'].
    - Demand: For each item i, the required demand comes from item_demand.csv column 'Demand'.
6.  **Formulate Objective:** Minimize the total number of standard rolls used, i.e., minimize the sum over all patterns p of y[p].
7.  **Formulate Constraints:**
    - Demand Satisfaction: For each item type i, the sum over all patterns p of (number of units of item i produced by pattern p) times y[p] must be greater than or equal to the demand for item i.
    - Nonnegativity and Integrality: For each pattern p, y[p] ≥ 0 and integer.
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum number of standard rolls to cut, by selecting how many times to use each feasible cutting pattern, such that the demand for each item type is satisfied or exceeded. The model must use integer variables for the number of times each pattern is used, ensure nonnegativity, and meet or exceed demand for all item types.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically an integer cutting-stock pattern-selection model.
3.  **Define Index Sets:** The primary indices are:
    - Patterns (from the 'Pattern' column in cutting_patterns.csv)
    - Item types (from the 'Item' column in item_demand.csv and the item columns in cutting_patterns.csv)
4.  **Define Decision Variables:**
    -   `y[p]` = Number of standard rolls cut using pattern p (for each pattern p in Patterns). Type: GRB.INTEGER.
5.  **Identify Parameters (from Schema):**
    -   Pattern composition: Number of units of each item type produced by each pattern, from columns 'A', 'B', 'C', 'D' in cutting_patterns.csv.
    -   Demand for each item type: From the 'Demand' column in item_demand.csv.
6.  **Formulate Objective:** Minimize the total number of standard rolls used, i.e., minimize the sum over all patterns of y[p].
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each item type i, the sum over all patterns p of (number of units of item i produced by pattern p) × y[p] must be greater than or equal to the demand for item i from item_demand.csv.
    -   Nonnegativity and Integrality: For each pattern p, y[p] ≥ 0 and integer.
[Abstract Model Plan END]
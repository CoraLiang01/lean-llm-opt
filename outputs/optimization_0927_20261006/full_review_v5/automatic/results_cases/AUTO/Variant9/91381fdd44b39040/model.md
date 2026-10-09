[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the minimum number of standard rolls to cut, by selecting how many times to use each feasible cutting pattern, such that the demand for each item type is satisfied or exceeded. Each pattern specifies how many units of each item type are produced per roll.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically an integer cutting-stock pattern-selection model.
3.  **Define Index Sets:** The primary indices are:
    - Item types (from item_demand.csv): set I (e.g., A, B, C, D)
    - Cutting patterns (from cutting_patterns.csv): set P (e.g., P1, P2, ..., P9)
4.  **Define Decision Variables:**
    -   `y[p]` = Number of standard rolls cut using pattern p ∈ P. Type: GRB.INTEGER (nonnegative integer).
5.  **Identify Parameters (from Schema):**
    -   Demand for each item type: from column 'Demand' in item_demand.csv, indexed by item i ∈ I.
    -   Units of each item type produced by each pattern: from columns 'A', 'B', 'C', 'D' in cutting_patterns.csv, indexed by pattern p ∈ P and item i ∈ I.
6.  **Formulate Objective:** Minimize the total number of standard rolls used, i.e., minimize the sum over all patterns of y[p].
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each item type i ∈ I, the total number of units produced across all patterns must meet or exceed the demand, i.e., sum over all patterns p of (units of item i produced by pattern p) × y[p] ≥ demand for item i.
    -   Nonnegativity and Integrality: For each pattern p ∈ P, y[p] ≥ 0 and integer.
[Abstract Model Plan END]
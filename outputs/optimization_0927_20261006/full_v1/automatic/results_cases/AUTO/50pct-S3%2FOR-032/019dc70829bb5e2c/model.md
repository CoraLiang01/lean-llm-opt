[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of university courses from the "Operations Research" discipline such that the total number of credits does not exceed 20, and the total "interest points" is maximized.
2.  **Identify Model Type:** Based on the query, this is a 0-1 Knapsack problem (a special case of Mixed Integer Programming, MIP).
3.  **Define Index Sets:** The primary index is the set of Operations Research courses, denoted as \( i \in \mathcal{OR} \), where each \( i \) corresponds to a row in the CSV with discipline = "Operations Research".
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if Operations Research course \( i \) is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'interest_points' column (the interest score for each course).
    -   Constraint coefficients: 'credits' column (the number of credits for each course).
    -   Constraint RHS: The total credit limit is 20 (a fixed value from the query, not from the data).
6.  **Formulate Objective:** Maximize the sum of 'interest_points' for selected Operations Research courses: maximize \(\sum_{i \in \mathcal{OR}} \text{interest_points}[i] \cdot x[i]\).
7.  **Formulate Constraints:**
    -   Credit Limit: The total credits of selected Operations Research courses must not exceed 20: \(\sum_{i \in \mathcal{OR}} \text{credits}[i] \cdot x[i] \leq 20\).
    -   Binary Selection: For each Operations Research course \( i \), \( x[i] \in \{0, 1\} \).
[Abstract Model Plan END]
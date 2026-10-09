[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of university courses from the "Operations Research" discipline such that the total number of credits does not exceed 20, and the sum of "interest points" is maximized.
2.  **Identify Model Type:** Based on the query, this is a 0-1 Knapsack Problem (a special case of Mixed Integer Programming, MIP).
3.  **Define Index Sets:** The primary index is the set of all courses in the CSV file where `discipline` equals "Operations Research". Let this set be denoted as $I$ (the set of eligible Operations Research courses).
4.  **Define Decision Variables:**
    -   `x[i]` = 1 if Operations Research course $i \in I$ is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: `interest_points` column (the interest score for each course).
    -   Constraint coefficients: `credits` column (the number of credits for each course).
    -   Constraint RHS: The total credit limit is 20 (fixed value from the query).
6.  **Formulate Objective:** Maximize the total interest points of selected Operations Research courses: $\max \sum_{i \in I} \text{interest_points}[i] \cdot x[i]$.
7.  **Formulate Constraints:**
    -   Credit Limit: $\sum_{i \in I} \text{credits}[i] \cdot x[i] \leq 20$ (the total credits of selected courses cannot exceed 20).
    -   Binary Selection: $x[i] \in \{0,1\}$ for all $i \in I$ (each course is either selected or not).
[Abstract Model Plan END]
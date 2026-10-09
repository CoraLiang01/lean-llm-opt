[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of university courses from the "Operations Research" discipline such that the total number of credits does not exceed 20, and the sum of their "interest points" is maximized.
2.  **Identify Model Type:** Based on the query, this is a 0-1 Knapsack Problem (a special case of Mixed Integer Programming, MIP).
3.  **Define Index Sets:** The primary index is the set of all courses in the CSV file where `discipline` = "Operations Research"; denote this set as $\mathcal{C}_{OR}$.
4.  **Define Decision Variables:**
    -   $x_i$ = 1 if course $i \in \mathcal{C}_{OR}$ is selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: `interest_points` column (for each course $i$).
    -   Constraint coefficients: `credits` column (for each course $i$).
    -   Constraint RHS: The total credit limit is 20 (fixed constant from the query).
6.  **Formulate Objective:** Maximize the total interest points of selected Operations Research courses: $\max \sum_{i \in \mathcal{C}_{OR}} \text{interest_points}_i \cdot x_i$.
7.  **Formulate Constraints:**
    -   Credit Limit: $\sum_{i \in \mathcal{C}_{OR}} \text{credits}_i \cdot x_i \leq 20$.
    -   Binary Selection: $x_i \in \{0,1\}$ for all $i \in \mathcal{C}_{OR}$.
[Abstract Model Plan END]
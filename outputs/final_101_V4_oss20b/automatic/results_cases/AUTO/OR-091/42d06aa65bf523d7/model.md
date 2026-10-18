[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of courses from the “Operations Research” discipline that maximizes total interest points while keeping the total number of credits no more than 20.
2.  **Identify Model Type:** This is a 0‑1 Integer Linear Programming (ILP) problem – a classic knapsack/subset‑selection model.
3.  **Define Index Sets:**  
    - `OR` – set of course indices `i` such that `discipline == 'Operations Research'` in `courses_42.csv`.
4.  **Define Decision Variables:**  
    - `x[i]` = 1 if course `i` is selected, 0 otherwise. Type: `GRB.BINARY`.
5.  **Identify Parameters (from Schema):**  
    - `credits[i]` – number of credits for course `i` (column `credits`).  
    - `interest_points[i]` – interest points for course `i` (column `interest_points`).  
    - `credit_limit` – maximum allowed credits, value `20`.
6.  **Formulate Objective:**  
    - `Maximize  Σ_{i∈OR} interest_points[i] * x[i]`
7.  **Formulate Constraints:**  
    - **Credit Limit:** `Σ_{i∈OR} credits[i] * x[i]  ≤  credit_limit`  
    - **Binary Decision:** `x[i] ∈ {0,1}  ∀ i ∈ OR`
[Abstract Model Plan END]
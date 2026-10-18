[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to develop a mathematical optimization model to determine the optimal daily production plan for a wine company that maximizes total net profit. The company produces three wine brands by blending three grades of raw wine, subject to blending proportion requirements, raw material supply limits, and a minimum production requirement for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Blending Linear Programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Brands (from 30-2.csv, column 'Brand')
    - Grades of raw wine (from 30-1.csv, column 'Grade')
4.  **Define Decision Variables:**
    -   `x[grade, brand]` = Quantity (kg) of raw wine of a given grade allocated to a given brand. Type: CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   Selling prices per brand: from 30-2.csv, column 'Selling Price (CNY/kg)'
    -   Blending requirements (proportion upper/lower bounds for certain grades in each brand): from 30-2.csv, column 'Blending Requirements'
    -   Raw material daily supply limits: from 30-1.csv, column 'Daily Supply (kg)'
    -   Raw material unit costs: from 30-1.csv, column 'Cost (CNY/kg)'
    -   Minimum production for Red brand: fixed at 2,000 kg (from query)
6.  **Formulate Objective:** Maximize total net profit, calculated as:  
    (Sum over all brands of [total quantity produced of each brand × its selling price])  
    minus  
    (Sum over all grades and brands of [quantity of each grade used in each brand × its unit cost]).
7.  **Formulate Constraints:**
    -   **Blending Requirements:** For each brand and each specified grade, the proportion of that grade in the total blend for the brand must satisfy the given upper or lower bound (e.g., for Red: proportion of Grade I < 10%, proportion of Grade II > 50%, etc.).
    -   **Raw Material Supply Limits:** For each grade, the total amount used across all brands cannot exceed its daily supply limit.
    -   **Minimum Production for Red Brand:** The total quantity produced of the Red brand (sum of all grades allocated to Red) must be at least 2,000 kg.
    -   **Non-negativity:** All decision variables must be greater than or equal to zero.
[Abstract Model Plan END]
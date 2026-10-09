[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), while satisfying blending proportion requirements, raw material supply limits, and a minimum production level for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a blending linear programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material (G): {I, II, III}
    - Wine brands (B): {Red, Yellow, Blue}
4.  **Define Decision Variables:**
    -   `x[g, b]` = amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    -   Raw material supply limits: from 30-1.csv, column 'Daily Supply (kg)' for each grade.
    -   Raw material costs: from 30-1.csv, column 'Cost (CNY/kg)' for each grade.
    -   Selling prices: from 30-2.csv, column 'Selling Price (CNY/kg)' for each brand.
    -   Blending requirements: from 30-2.csv, column 'Blending Requirements' for each brand (specifying upper/lower bounds on proportions of certain grades in each brand).
    -   Minimum production for Red: fixed at 2,000 kg.
6.  **Formulate Objective:** Maximize total net profit, calculated as the sum over all brands of (total amount produced of each brand × its selling price) minus the sum over all grades and brands of (amount of each grade used × its cost).
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand and each specified grade, the proportion of that grade in the total blend for the brand must satisfy the given upper or lower bound (e.g., for Red: proportion of I < 10%, proportion of II > 50%, etc.).
    -   Constraint 2 (Raw Material Supply): For each grade, the total amount used across all brands cannot exceed its daily supply limit.
    -   Constraint 3 (Minimum Production for Red): The total amount produced of the Red brand (sum of all grades used in Red) must be at least 2,000 kg.
    -   Constraint 4 (Non-negativity): All decision variables `x[g, b]` must be greater than or equal to zero.
[Abstract Model Plan END]
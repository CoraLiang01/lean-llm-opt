[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), subject to blending proportion requirements for each brand, raw material supply limits, and a minimum production requirement for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material (G): {I, II, III}
    - Wine brands (B): {Red, Yellow, Blue}
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    -   Raw material supply limits and costs from 30-1.csv: 'Grade', 'Daily Supply (kg)', 'Cost (CNY/kg)'
    -   Brand selling prices and blending requirements from 30-2.csv: 'Brand', 'Blending Requirements', 'Selling Price (CNY/kg)'
6.  **Formulate Objective:** Maximize total net profit, calculated as the sum over all brands of (total amount produced of each brand × its selling price) minus the sum over all grades of (total amount of each grade used × its unit cost).
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand, enforce upper and/or lower bounds on the proportion of selected grades in the blend, as specified in 'Blending Requirements' (e.g., for Red: proportion of Grade I < 10%, proportion of Grade II > 50%; for Yellow: proportion of Grade III < 70%, Grade I > 20%; for Blue: Grade I < 50%, Grade II > 10%). For each brand b and relevant grade g, the constraint is: x[g, b] / sum_{g'} x[g', b] ≤ or ≥ specified proportion.
    -   Constraint 2 (Raw Material Supply): For each grade, the total amount used across all brands cannot exceed its daily supply limit: sum_{b} x[g, b] ≤ supply limit for grade g.
    -   Constraint 3 (Minimum Production for Red): The total daily production of the Red brand must be at least 2,000 kg: sum_{g} x[g, Red] ≥ 2,000.
    -   Constraint 4 (Non-negativity): All decision variables x[g, b] ≥ 0.
[Abstract Model Plan END]
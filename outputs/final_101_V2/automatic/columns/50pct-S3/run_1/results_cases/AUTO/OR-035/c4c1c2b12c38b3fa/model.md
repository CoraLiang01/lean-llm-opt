[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum for fat. All nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending/diet problem.
3.  **Define Index Sets:** The primary index is Foods (i), where each food is a row in the CSV file.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings of food i to include in the meal plan. Type: GRB.CONTINUOUS (can be fractional, ≥ 0).
5.  **Identify Parameters (from Schema):**
    -   Cost per serving: from column 'Cost'.
    -   Calories per serving: from column 'Calories'.
    -   Protein per serving: from column 'Protein(g)'.
    -   Fat per serving: from column 'Fat(g)'.
    -   Vitamin C per serving: from column 'VitaminC(mg)'.
    -   All foods (120 rows) are included; no filtering is required.
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize sum over all foods of (Cost[i] * x[i]).
7.  **Formulate Constraints:**
    -   Calorie minimum: sum over all foods of (Calories[i] * x[i]) ≥ 2000.
    -   Protein minimum: sum over all foods of (Protein(g)[i] * x[i]) ≥ 50.
    -   Vitamin C minimum: sum over all foods of (VitaminC(mg)[i] * x[i]) ≥ 60.
    -   Fat maximum: sum over all foods of (Fat(g)[i] * x[i]) ≤ 70.
    -   Non-negativity: x[i] ≥ 0 for all foods i.
[Abstract Model Plan END]
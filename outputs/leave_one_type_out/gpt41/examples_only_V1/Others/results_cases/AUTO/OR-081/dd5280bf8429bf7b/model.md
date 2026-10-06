[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum for fat. All nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) problem (continuous variables, linear objective and constraints).
3.  **Define Index Sets:** The primary index is Foods (i), where i ranges over all 120 foods listed in the CSV.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings of food i to include in the meal plan. Type: GRB.CONTINUOUS (can be fractional, ≥ 0).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving of each food).
    -   Constraint coefficients: 'Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)' columns (nutrient content per serving).
    -   Constraint RHS (limits): 2000 (min Calories), 50 (min Protein), 60 (min Vitamin C), 70 (max Fat).
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize sum over all foods of (Cost[i] * x[i]).
7.  **Formulate Constraints:**
    -   Constraint 1 (Calorie Minimum): The sum over all foods of (Calories[i] * x[i]) ≥ 2000.
    -   Constraint 2 (Protein Minimum): The sum over all foods of (Protein(g)[i] * x[i]) ≥ 50.
    -   Constraint 3 (Vitamin C Minimum): The sum over all foods of (VitaminC(mg)[i] * x[i]) ≥ 60.
    -   Constraint 4 (Fat Maximum): The sum over all foods of (Fat(g)[i] * x[i]) ≤ 70.
    -   Constraint 5 (Non-negativity): For all foods i, x[i] ≥ 0.
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum for fat. All nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is Foods (let the set of foods be indexed by i).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings of food i to include in the meal plan. Type: GRB.CONTINUOUS (x[i] ≥ 0; fractional servings allowed).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving for each food).
    -   Constraint coefficients: 'Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)' columns (nutrient content per serving for each food).
    -   Constraint RHS: 2000 (minimum calories), 50 (minimum protein in grams), 60 (minimum vitamin C in mg), 70 (maximum fat in grams).
6.  **Formulate Objective:** Minimize total cost: sum over all foods of (Cost[i] * x[i]).
7.  **Formulate Constraints:**
    -   Calorie constraint: sum over all foods of (Calories[i] * x[i]) ≥ 2000.
    -   Protein constraint: sum over all foods of (Protein(g)[i] * x[i]) ≥ 50.
    -   Vitamin C constraint: sum over all foods of (VitaminC(mg)[i] * x[i]) ≥ 60.
    -   Fat constraint: sum over all foods of (Fat(g)[i] * x[i]) ≤ 70.
    -   Non-negativity: x[i] ≥ 0 for all foods i.
[Abstract Model Plan END]
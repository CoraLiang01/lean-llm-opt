[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while ensuring the meal plan meets specific nutritional requirements: at least 2000 Calories, at least 50 g of protein, at least 60 mg of vitamin C, and no more than 70 g of fat.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is the set of available foods, denoted as `i ∈ Foods` (where Foods is all 120 rows in the CSV).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings (can be fractional) of food `i` to include in the meal plan. Type: GRB.CONTINUOUS, with `x[i] ≥ 0`.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving of food `i`).
    -   Constraint coefficients:
        -   'Calories' (kcal per serving of food `i`)
        -   'Protein(g)' (grams of protein per serving)
        -   'Fat(g)' (grams of fat per serving)
        -   'VitaminC(mg)' (mg of vitamin C per serving)
    -   Constraint RHS (limits): Provided in the query (Calories ≥ 2000, Protein ≥ 50 g, Vitamin C ≥ 60 mg, Fat ≤ 70 g).
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize sum over all foods of (`Cost[i]` * `x[i]`).
7.  **Formulate Constraints:**
    -   Constraint 1 (Calories): The total calories from all selected foods must be at least 2000: sum over all foods of (`Calories[i]` * `x[i]`) ≥ 2000.
    -   Constraint 2 (Protein): The total protein must be at least 50 g: sum over all foods of (`Protein(g)[i]` * `x[i]`) ≥ 50.
    -   Constraint 3 (Vitamin C): The total vitamin C must be at least 60 mg: sum over all foods of (`VitaminC(mg)[i]` * `x[i]`) ≥ 60.
    -   Constraint 4 (Fat): The total fat must not exceed 70 g: sum over all foods of (`Fat(g)[i]` * `x[i]`) ≤ 70.
    -   Constraint 5 (Non-negativity): For all foods, `x[i]` ≥ 0.
[Abstract Model Plan END]
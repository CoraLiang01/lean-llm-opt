[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum for fat. All relevant nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is Foods (set of all foods listed in the CSV).
4.  **Define Decision Variables:**
    -   `x[f]` = Number of servings of food `f` to include in the meal plan. Type: GRB.CONTINUOUS (allowing fractional servings), with lower bound 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving of each food).
    -   Constraint coefficients:
        -   'Calories' (calories per serving)
        -   'Protein(g)' (protein per serving)
        -   'VitaminC(mg)' (vitamin C per serving)
        -   'Fat(g)' (fat per serving)
    -   Constraint RHS (limits):
        -   Calories: at least 2000
        -   Protein: at least 50 g
        -   Vitamin C: at least 60 mg
        -   Fat: no more than 70 g
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize sum over all foods of (Cost per serving) × (number of servings selected): minimize ∑₍f₎ Cost[f] * x[f].
7.  **Formulate Constraints:**
    -   Calorie constraint: The sum of calories from all selected servings must be at least 2000: ∑₍f₎ Calories[f] * x[f] ≥ 2000.
    -   Protein constraint: The sum of protein from all selected servings must be at least 50 g: ∑₍f₎ Protein(g)[f] * x[f] ≥ 50.
    -   Vitamin C constraint: The sum of vitamin C from all selected servings must be at least 60 mg: ∑₍f₎ VitaminC(mg)[f] * x[f] ≥ 60.
    -   Fat constraint: The sum of fat from all selected servings must not exceed 70 g: ∑₍f₎ Fat(g)[f] * x[f] ≤ 70.
    -   Non-negativity: For all foods, x[f] ≥ 0.
[Abstract Model Plan END]
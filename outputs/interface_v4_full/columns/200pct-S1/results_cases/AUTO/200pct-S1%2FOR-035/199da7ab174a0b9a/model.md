[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum for fat. All nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending/diet problem.
3.  **Define Index Sets:** The primary index is Foods (i ∈ set of all foods listed in the CSV; 120 foods).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings of food i to include in the meal plan. Type: GRB.CONTINUOUS (can be fractional, x[i] ≥ 0).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving of food i).
    -   Constraint coefficients:
        -   'Calories' (kcal per serving of food i)
        -   'Protein(g)' (grams of protein per serving of food i)
        -   'Fat(g)' (grams of fat per serving of food i)
        -   'VitaminC(mg)' (mg of vitamin C per serving of food i)
    -   Constraint RHS (limits):
        -   Calories: at least 2000 kcal
        -   Protein: at least 50 g
        -   Vitamin C: at least 60 mg
        -   Fat: no more than 70 g
6.  **Formulate Objective:** Minimize the total cost of the meal plan: sum over all foods of (Cost per serving) × (number of servings), i.e., minimize ∑₍ᵢ₎ Cost[i] * x[i].
7.  **Formulate Constraints:**
    -   Calorie constraint: sum over all foods of (Calories per serving) × (number of servings) ≥ 2000, i.e., ∑₍ᵢ₎ Calories[i] * x[i] ≥ 2000.
    -   Protein constraint: sum over all foods of (Protein per serving) × (number of servings) ≥ 50, i.e., ∑₍ᵢ₎ Protein[i] * x[i] ≥ 50.
    -   Vitamin C constraint: sum over all foods of (VitaminC per serving) × (number of servings) ≥ 60, i.e., ∑₍ᵢ₎ VitaminC[i] * x[i] ≥ 60.
    -   Fat constraint: sum over all foods of (Fat per serving) × (number of servings) ≤ 70, i.e., ∑₍ᵢ₎ Fat[i] * x[i] ≤ 70.
    -   Non-negativity: x[i] ≥ 0 for all foods i.
[Abstract Model Plan END]
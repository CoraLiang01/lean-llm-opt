[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum fat limit. All per-serving nutrient and cost data are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is Foods (each row in the cost.csv table).
4.  **Define Decision Variables:**
    -   `x[f]` = Number of servings (can be fractional) of food `f` to include in the meal plan. Type: GRB.CONTINUOUS, with `x[f] >= 0`.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving for each food).
    -   Constraint coefficients:
        -   'Calories' (kcal per serving)
        -   'Protein(g)' (grams per serving)
        -   'Fat(g)' (grams per serving)
        -   'VitaminC(mg)' (mg per serving)
    -   Constraint RHS (limits):
        -   Calories: at least 2000
        -   Protein: at least 50 g
        -   Vitamin C: at least 60 mg
        -   Fat: no more than 70 g
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize the sum over all foods of (Cost per serving) × (number of servings chosen):  
    Minimize: sum over all foods f of `Cost[f] * x[f]`
7.  **Formulate Constraints:**
    -   Calorie constraint: sum over all foods f of `Calories[f] * x[f]` ≥ 2000
    -   Protein constraint: sum over all foods f of `Protein(g)[f] * x[f]` ≥ 50
    -   Vitamin C constraint: sum over all foods f of `VitaminC(mg)[f] * x[f]` ≥ 60
    -   Fat constraint: sum over all foods f of `Fat(g)[f] * x[f]` ≤ 70
    -   Non-negativity: For all foods f, `x[f] ≥ 0`
[Abstract Model Plan END]
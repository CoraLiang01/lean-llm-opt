[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum for fat. All per-serving nutrient and cost data are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is the set of Foods, as listed in the 'Food' column of cost.csv (all rows included).
4.  **Define Decision Variables:**
    -   `x[f]` = Number of servings (can be fractional) of food item `f` to include in the meal plan. Type: GRB.CONTINUOUS, with lower bound 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving for each food).
    -   Constraint coefficients:
        -   'Calories' (kcal per serving)
        -   'Protein(g)' (grams per serving)
        -   'Fat(g)' (grams per serving)
        -   'VitaminC(mg)' (mg per serving)
    -   Constraint right-hand sides (RHS):
        -   Calories: at least 2000
        -   Protein: at least 50
        -   Vitamin C: at least 60
        -   Fat: no more than 70
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize the sum over all foods of (Cost per serving) × (number of servings selected):  
    Minimize:  sum_f [ Cost[f] * x[f] ]
7.  **Formulate Constraints:**
    -   Calorie requirement: sum_f [ Calories[f] * x[f] ] ≥ 2000
    -   Protein requirement: sum_f [ Protein(g)[f] * x[f] ] ≥ 50
    -   Vitamin C requirement: sum_f [ VitaminC(mg)[f] * x[f] ] ≥ 60
    -   Fat limit: sum_f [ Fat(g)[f] * x[f] ] ≤ 70
    -   Non-negativity: x[f] ≥ 0 for all foods f
[Abstract Model Plan END]
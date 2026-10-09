[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of various foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum for fat. All per-serving nutrient and cost data are provided in cost.csv.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending/diet problem.
3.  **Define Index Sets:** The primary index is Foods (i), where each food is a row in cost.csv (all 120 foods are included).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings of food i to include in the meal plan. Type: GRB.CONTINUOUS (can be fractional, x[i] ≥ 0).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving of food i).
    -   Constraint coefficients:
        -   'Calories' (kcal per serving of food i)
        -   'Protein(g)' (grams of protein per serving)
        -   'Fat(g)' (grams of fat per serving)
        -   'VitaminC(mg)' (mg of vitamin C per serving)
    -   Constraint RHS (limits):
        -   Calories: at least 2000 kcal
        -   Protein: at least 50 g
        -   Vitamin C: at least 60 mg
        -   Fat: no more than 70 g
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize sum over all foods i of (Cost[i] * x[i]).
7.  **Formulate Constraints:**
    -   Calorie constraint: sum over i of (Calories[i] * x[i]) ≥ 2000
    -   Protein constraint: sum over i of (Protein(g)[i] * x[i]) ≥ 50
    -   Vitamin C constraint: sum over i of (VitaminC(mg)[i] * x[i]) ≥ 60
    -   Fat constraint: sum over i of (Fat(g)[i] * x[i]) ≤ 70
    -   Non-negativity: x[i] ≥ 0 for all foods i
[Abstract Model Plan END]
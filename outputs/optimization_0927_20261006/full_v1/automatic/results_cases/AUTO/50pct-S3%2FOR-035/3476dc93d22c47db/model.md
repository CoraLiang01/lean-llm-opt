[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum fat limit. All nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending/diet problem.
3.  **Define Index Sets:** The primary index is the set of Foods, denoted as 𝑖 ∈ Foods (all rows in cost.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings of food i to include in the meal plan. Type: GRB.CONTINUOUS (x[i] ≥ 0, fractional servings allowed).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving for each food).
    -   Constraint coefficients:
        -   Calories per serving: 'Calories'
        -   Protein per serving: 'Protein(g)'
        -   Fat per serving: 'Fat(g)'
        -   Vitamin C per serving: 'VitaminC(mg)'
    -   Constraint RHS (limits):
        -   Calories: at least 2000
        -   Protein: at least 50
        -   Vitamin C: at least 60
        -   Fat: no more than 70
6.  **Formulate Objective:** Minimize the total cost of the meal plan: minimize sum over i of (Cost[i] * x[i]).
7.  **Formulate Constraints:**
    -   Calorie requirement: sum over i of (Calories[i] * x[i]) ≥ 2000
    -   Protein requirement: sum over i of (Protein(g)[i] * x[i]) ≥ 50
    -   Vitamin C requirement: sum over i of (VitaminC(mg)[i] * x[i]) ≥ 60
    -   Fat limit: sum over i of (Fat(g)[i] * x[i]) ≤ 70
    -   Non-negativity: x[i] ≥ 0 for all i ∈ Foods
[Abstract Model Plan END]
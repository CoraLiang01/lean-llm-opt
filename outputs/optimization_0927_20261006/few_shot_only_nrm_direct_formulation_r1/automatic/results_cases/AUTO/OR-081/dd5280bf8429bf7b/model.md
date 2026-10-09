[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum for fat. All per-serving nutrient and cost data are provided in cost.csv.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is the set of Foods, denoted as `i ∈ Foods`, where Foods are all rows in cost.csv.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings of food `i` to include in the meal plan. Type: GRB.CONTINUOUS (non-negative, can be fractional).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: `Cost` column in cost.csv gives the cost per serving for each food.
    -   Constraint coefficients:
        -   Calories per serving: `Calories`
        -   Protein per serving: `Protein(g)`
        -   Fat per serving: `Fat(g)`
        -   Vitamin C per serving: `VitaminC(mg)`
    -   Constraint RHS (limits):
        -   Calories: at least 2000 kcal
        -   Protein: at least 50 g
        -   Vitamin C: at least 60 mg
        -   Fat: no more than 70 g
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize the sum over all foods of (Cost per serving) × (number of servings):  
    Minimize:  sum over i of [Cost[i] * x[i]]
7.  **Formulate Constraints:**
    -   Calorie constraint: sum over i of [Calories[i] * x[i]] ≥ 2000
    -   Protein constraint: sum over i of [Protein(g)[i] * x[i]] ≥ 50
    -   Vitamin C constraint: sum over i of [VitaminC(mg)[i] * x[i]] ≥ 60
    -   Fat constraint: sum over i of [Fat(g)[i] * x[i]] ≤ 70
    -   Non-negativity: x[i] ≥ 0 for all i ∈ Foods
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while ensuring the meal plan meets minimum requirements for calories, protein, and vitamin C, and does not exceed a maximum fat limit. All relevant nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is the set of Foods (i ∈ Foods), as listed in the 'Food' column of the CSV.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings (can be fractional) of food i to include in the meal plan. Type: GRB.CONTINUOUS, with x[i] ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving of food i).
    -   Constraint coefficients:
        -   Calories per serving: 'Calories'
        -   Protein per serving: 'Protein(g)'
        -   Fat per serving: 'Fat(g)'
        -   Vitamin C per serving: 'VitaminC(mg)'
    -   Constraint RHS (limits):
        -   Calories: at least 2000
        -   Protein: at least 50 g
        -   Vitamin C: at least 60 mg
        -   Fat: no more than 70 g
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize sum over all foods i of (schema['Cost'][i] * x[i]).
7.  **Formulate Constraints:**
    -   Calorie constraint: sum over i of (schema['Calories'][i] * x[i]) ≥ 2000
    -   Protein constraint: sum over i of (schema['Protein(g)'][i] * x[i]) ≥ 50
    -   Vitamin C constraint: sum over i of (schema['VitaminC(mg)'][i] * x[i]) ≥ 60
    -   Fat constraint: sum over i of (schema['Fat(g)'][i] * x[i]) ≤ 70
    -   Non-negativity: x[i] ≥ 0 for all i ∈ Foods
[Abstract Model Plan END]
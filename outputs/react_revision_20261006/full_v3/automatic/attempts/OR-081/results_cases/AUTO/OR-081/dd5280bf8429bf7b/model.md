[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while ensuring the meal plan meets specified nutritional requirements: at least 2000 Calories, at least 50 g protein, at least 60 mg vitamin C, and no more than 70 g fat. All per-serving nutrient and cost data are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is the set of Foods, denoted as `i`, where each `i` corresponds to a row in the CSV (all 120 foods).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings (can be fractional) of food `i` to include in the meal plan. Type: GRB.CONTINUOUS (non-negative).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving for each food).
    -   Constraint coefficients:
        -   'Calories' (kcal per serving)
        -   'Protein(g)' (grams per serving)
        -   'Fat(g)' (grams per serving)
        -   'VitaminC(mg)' (mg per serving)
    -   Constraint RHS (limits):
        -   Calories: minimum 2000
        -   Protein: minimum 50 g
        -   Vitamin C: minimum 60 mg
        -   Fat: maximum 70 g
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize the sum over all foods of (cost per serving) × (number of servings):  
    Minimize: sum over i of `Cost[i] * x[i]`
7.  **Formulate Constraints:**
    -   Constraint 1 (Calorie Minimum): The total calories from all selected foods must be at least 2000.  
        sum over i of `Calories[i] * x[i]` ≥ 2000
    -   Constraint 2 (Protein Minimum): The total protein must be at least 50 g.  
        sum over i of `Protein(g)[i] * x[i]` ≥ 50
    -   Constraint 3 (Vitamin C Minimum): The total vitamin C must be at least 60 mg.  
        sum over i of `VitaminC(mg)[i] * x[i]` ≥ 60
    -   Constraint 4 (Fat Maximum): The total fat must not exceed 70 g.  
        sum over i of `Fat(g)[i] * x[i]` ≤ 70
    -   Constraint 5 (Non-negativity):  
        For all i, `x[i]` ≥ 0
[Abstract Model Plan END]
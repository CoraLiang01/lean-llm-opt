[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while ensuring the meal plan meets minimum requirements for calories, protein, and vitamin C, and does not exceed a maximum fat limit. All nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is the set of available foods, indexed by `i` (i.e., all rows in the 'Food' column of the CSV).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings (can be fractional) of food `i` to include in the meal plan. Type: GRB.CONTINUOUS, with lower bound 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving of food `i`).
    -   Constraint coefficients:
        -   'Calories' (calories per serving of food `i`)
        -   'Protein(g)' (protein per serving of food `i`)
        -   'Fat(g)' (fat per serving of food `i`)
        -   'VitaminC(mg)' (vitamin C per serving of food `i`)
    -   Constraint right-hand sides (RHS):
        -   Calories: at least 2000
        -   Protein: at least 50
        -   Vitamin C: at least 60
        -   Fat: no more than 70
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize the sum over all foods of (Cost per serving) × (number of servings):  
    Minimize ∑₍ᵢ₎ Cost[i] × x[i]
7.  **Formulate Constraints:**
    -   Calorie constraint: The total calories from all selected servings must be at least 2000:  
        ∑₍ᵢ₎ Calories[i] × x[i] ≥ 2000
    -   Protein constraint: The total protein from all selected servings must be at least 50g:  
        ∑₍ᵢ₎ Protein(g)[i] × x[i] ≥ 50
    -   Vitamin C constraint: The total vitamin C from all selected servings must be at least 60mg:  
        ∑₍ᵢ₎ VitaminC(mg)[i] × x[i] ≥ 60
    -   Fat constraint: The total fat from all selected servings must not exceed 70g:  
        ∑₍ᵢ₎ Fat(g)[i] × x[i] ≤ 70
    -   Non-negativity: For all foods, x[i] ≥ 0
[Abstract Model Plan END]
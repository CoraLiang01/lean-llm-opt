[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while ensuring the meal plan meets minimum requirements for calories, protein, and vitamin C, and does not exceed a maximum fat limit. All nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is Foods (i), where each food item is a row in the CSV file.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings (can be fractional) of food i to include in the meal plan. Type: GRB.CONTINUOUS, with lower bound 0 (no negative servings).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving of food i).
    -   Constraint coefficients:
        -   'Calories' (calories per serving of food i)
        -   'Protein(g)' (protein per serving of food i)
        -   'Fat(g)' (fat per serving of food i)
        -   'VitaminC(mg)' (vitamin C per serving of food i)
    -   Constraint RHS (limits): Provided in the query:
        -   Calories: at least 2000 kcal
        -   Protein: at least 50 g
        -   Vitamin C: at least 60 mg
        -   Fat: no more than 70 g
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize the sum over all foods of (Cost per serving) × (number of servings chosen):  
    Minimize  ∑₍ᵢ₎  Cost[i] × x[i]
7.  **Formulate Constraints:**
    -   Constraint 1 (Calorie Minimum): The total calories from all selected foods must be at least 2000 kcal:  
        ∑₍ᵢ₎  Calories[i] × x[i]  ≥  2000
    -   Constraint 2 (Protein Minimum): The total protein from all selected foods must be at least 50 g:  
        ∑₍ᵢ₎  Protein(g)[i] × x[i]  ≥  50
    -   Constraint 3 (Vitamin C Minimum): The total vitamin C from all selected foods must be at least 60 mg:  
        ∑₍ᵢ₎  VitaminC(mg)[i] × x[i]  ≥  60
    -   Constraint 4 (Fat Maximum): The total fat from all selected foods must not exceed 70 g:  
        ∑₍ᵢ₎  Fat(g)[i] × x[i]  ≤  70
    -   Constraint 5 (Non-negativity):  
        For all i,  x[i]  ≥  0
[Abstract Model Plan END]
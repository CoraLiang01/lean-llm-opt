[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings of available foods to minimize total cost, while ensuring the meal plan meets specified nutritional requirements: at least 2000 Calories, at least 50 g protein, at least 60 mg vitamin C, and no more than 70 g fat. Fractional servings are allowed.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is Foods (i), where each food is a row in the CSV file (all 120 foods).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings of food i to include in the meal plan. Type: GRB.CONTINUOUS (can be fractional, ≥ 0).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving of food i).
    -   Constraint coefficients:
        -   'Calories' (kcal per serving of food i)
        -   'Protein(g)' (grams of protein per serving)
        -   'Fat(g)' (grams of fat per serving)
        -   'VitaminC(mg)' (mg of vitamin C per serving)
    -   Constraint RHS (limits):
        -   Calories: minimum 2000
        -   Protein: minimum 50 g
        -   Vitamin C: minimum 60 mg
        -   Fat: maximum 70 g
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize the sum over all foods of (Cost per serving) × (number of servings chosen):  
    Minimize  ∑₍ᵢ₎  Cost[i] × x[i]
7.  **Formulate Constraints:**
    -   Constraint 1 (Calorie Minimum):  
        ∑₍ᵢ₎  Calories[i] × x[i]  ≥ 2000
    -   Constraint 2 (Protein Minimum):  
        ∑₍ᵢ₎  Protein(g)[i] × x[i]  ≥ 50
    -   Constraint 3 (Vitamin C Minimum):  
        ∑₍ᵢ₎  VitaminC(mg)[i] × x[i]  ≥ 60
    -   Constraint 4 (Fat Maximum):  
        ∑₍ᵢ₎  Fat(g)[i] × x[i]  ≤ 70
    -   Constraint 5 (Non-negativity):  
        x[i] ≥ 0  for all foods i
[Abstract Model Plan END]
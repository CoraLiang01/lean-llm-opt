[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum for fat. All nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending/diet problem.
3.  **Define Index Sets:** The primary index is Foods (i), where i ranges over all 120 foods listed in the CSV file.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings of food i to include in the meal plan. Type: GRB.CONTINUOUS (can be fractional, ≥ 0).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving of food i).
    -   Constraint coefficients:
        -   'Calories' (kcal per serving of food i)
        -   'Protein(g)' (grams per serving of food i)
        -   'VitaminC(mg)' (mg per serving of food i)
        -   'Fat(g)' (grams per serving of food i)
    -   Constraint RHS (limits):
        -   Calories: at least 2000 kcal
        -   Protein: at least 50 g
        -   Vitamin C: at least 60 mg
        -   Fat: no more than 70 g
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize sum over all foods of (Cost per serving) × (number of servings):  
    Minimize  ∑₍ᵢ₎  Cost[i] × x[i]
7.  **Formulate Constraints:**
    -   Constraint 1 (Calories minimum):  
        ∑₍ᵢ₎  Calories[i] × x[i]  ≥ 2000
    -   Constraint 2 (Protein minimum):  
        ∑₍ᵢ₎  Protein(g)[i] × x[i]  ≥ 50
    -   Constraint 3 (Vitamin C minimum):  
        ∑₍ᵢ₎  VitaminC(mg)[i] × x[i]  ≥ 60
    -   Constraint 4 (Fat maximum):  
        ∑₍ᵢ₎  Fat(g)[i] × x[i]  ≤ 70
    -   Constraint 5 (Non-negativity):  
        x[i] ≥ 0  for all foods i
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum for fat. All per-serving nutrient and cost data are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is the set of Foods, as listed in the 'Food' column of cost.csv. All rows are required; no filtering is specified.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings (can be fractional) of food item `i` to include in the meal plan. Type: GRB.CONTINUOUS, with lower bound 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (USD per serving).
    -   Constraint coefficients:
        -   'Calories' (kcal per serving)
        -   'Protein(g)' (grams per serving)
        -   'Fat(g)' (grams per serving)
        -   'VitaminC(mg)' (mg per serving)
    -   Constraint right-hand sides (RHS):
        -   Calories: at least 2000 kcal
        -   Protein: at least 50 g
        -   Vitamin C: at least 60 mg
        -   Fat: no more than 70 g
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize the sum over all foods of (Cost per serving) × (number of servings selected):  
    Minimize  ∑₍ᵢ₎  Cost[i] × x[i]
7.  **Formulate Constraints:**
    -   Calorie requirement:  ∑₍ᵢ₎  Calories[i] × x[i]  ≥  2000
    -   Protein requirement:  ∑₍ᵢ₎  Protein[i] × x[i]  ≥  50
    -   Vitamin C requirement:  ∑₍ᵢ₎  VitaminC[i] × x[i]  ≥  60
    -   Fat limit:  ∑₍ᵢ₎  Fat[i] × x[i]  ≤  70
    -   Non-negativity:  x[i]  ≥  0  for all foods i
[Abstract Model Plan END]
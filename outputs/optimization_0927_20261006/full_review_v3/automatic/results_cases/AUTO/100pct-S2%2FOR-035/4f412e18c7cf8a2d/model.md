[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum for fat. All nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending/diet problem.
3.  **Define Index Sets:** The primary index is Foods (set of all foods listed in the CSV).
4.  **Define Decision Variables:**
    -   `x[f]` = Number of servings of food `f` to include in the meal plan. Type: GRB.CONTINUOUS (allowing fractional servings), with lower bound 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving for each food).
    -   Constraint coefficients:
        -   'Calories' (kcal per serving)
        -   'Protein(g)' (grams per serving)
        -   'VitaminC(mg)' (mg per serving)
        -   'Fat(g)' (grams per serving)
    -   Constraint RHS (limits):
        -   Calories: minimum 2000
        -   Protein: minimum 50
        -   Vitamin C: minimum 60
        -   Fat: maximum 70
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize sum over all foods of (Cost per serving) × (number of servings):  
    Minimize  ∑₍f∈Foods₎  Cost[f] × x[f]
7.  **Formulate Constraints:**
    -   Calorie constraint:  ∑₍f∈Foods₎  Calories[f] × x[f]  ≥ 2000
    -   Protein constraint:  ∑₍f∈Foods₎  Protein(g)[f] × x[f]  ≥ 50
    -   Vitamin C constraint:  ∑₍f∈Foods₎  VitaminC(mg)[f] × x[f]  ≥ 60
    -   Fat constraint:  ∑₍f∈Foods₎  Fat(g)[f] × x[f]  ≤ 70
    -   Non-negativity:  x[f] ≥ 0  for all foods f
[Abstract Model Plan END]
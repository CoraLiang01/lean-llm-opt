[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum fat limit. All per-serving nutrient and cost data are provided in cost.csv, and fractional servings are allowed.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is Foods (set of all foods listed in cost.csv).
4.  **Define Decision Variables:**
    -   `x[f]` = Number of servings of food `f` to include in the meal plan. Type: GRB.CONTINUOUS (x[f] ≥ 0, fractional values allowed).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column in cost.csv (cost per serving for each food).
    -   Constraint coefficients: 'Calories', 'Protein(g)', 'Fat(g)', 'VitaminC(mg)' columns in cost.csv (per-serving nutrient content).
    -   Constraint RHS: 2000 (minimum calories), 50 (minimum protein in grams), 60 (minimum vitamin C in mg), 70 (maximum fat in grams).
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize the sum over all foods of (Cost per serving) × (number of servings selected):  
    Minimize ∑₍f∈Foods₎ Cost[f] × x[f].
7.  **Formulate Constraints:**
    -   Calorie constraint: The total calories from all selected foods must be at least 2000.  
        ∑₍f∈Foods₎ Calories[f] × x[f] ≥ 2000.
    -   Protein constraint: The total protein from all selected foods must be at least 50 grams.  
        ∑₍f∈Foods₎ Protein[g][f] × x[f] ≥ 50.
    -   Vitamin C constraint: The total vitamin C from all selected foods must be at least 60 mg.  
        ∑₍f∈Foods₎ VitaminC[mg][f] × x[f] ≥ 60.
    -   Fat constraint: The total fat from all selected foods must not exceed 70 grams.  
        ∑₍f∈Foods₎ Fat[g][f] × x[f] ≤ 70.
    -   Non-negativity: For all foods f, x[f] ≥ 0.
[Abstract Model Plan END]
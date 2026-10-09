[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum for fat. All nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is the set of Foods (i ∈ Foods), as listed in the 'Food' column of cost.csv.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings (can be fractional) of food i to include in the meal plan. Type: GRB.CONTINUOUS, x[i] ≥ 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving for each food).
    -   Constraint coefficients:
        -   'Calories' (kcal per serving)
        -   'Protein(g)' (grams per serving)
        -   'VitaminC(mg)' (mg per serving)
        -   'Fat(g)' (grams per serving)
    -   Constraint right-hand sides (RHS):
        -   Calories: at least 2000
        -   Protein: at least 50
        -   Vitamin C: at least 60
        -   Fat: no more than 70
6.  **Formulate Objective:** Minimize the total cost of the meal plan: sum over all foods of (Cost per serving) × (number of servings), i.e., minimize ∑₍ᵢ∈Foods₎ Cost[i] × x[i].
7.  **Formulate Constraints:**
    -   Calorie constraint: ∑₍ᵢ∈Foods₎ Calories[i] × x[i] ≥ 2000
    -   Protein constraint: ∑₍ᵢ∈Foods₎ Protein(g)[i] × x[i] ≥ 50
    -   Vitamin C constraint: ∑₍ᵢ∈Foods₎ VitaminC(mg)[i] × x[i] ≥ 60
    -   Fat constraint: ∑₍ᵢ∈Foods₎ Fat(g)[i] × x[i] ≤ 70
    -   Non-negativity: x[i] ≥ 0 for all i ∈ Foods
[Abstract Model Plan END]
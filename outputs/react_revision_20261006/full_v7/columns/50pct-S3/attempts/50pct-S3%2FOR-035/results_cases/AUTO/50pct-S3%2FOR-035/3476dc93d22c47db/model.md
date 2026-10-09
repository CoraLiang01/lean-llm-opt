[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum for fat. All nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending/diet problem.
3.  **Define Index Sets:** The primary index is the set of Foods, denoted as `i`, where each `i` corresponds to a row in the cost.csv file (all 120 foods).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings (can be fractional) of food `i` to include in the meal plan. Type: GRB.CONTINUOUS, with lower bound 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: `Cost` column (cost per serving in USD).
    -   Constraint coefficients:
        -   Calories per serving: `Calories`
        -   Protein per serving: `Protein(g)`
        -   Fat per serving: `Fat(g)`
        -   Vitamin C per serving: `VitaminC(mg)`
    -   Constraint RHS (limits):
        -   Calories: at least 2000 kcal
        -   Protein: at least 50 g
        -   Vitamin C: at least 60 mg
        -   Fat: no more than 70 g
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize the sum over all foods of (`Cost` per serving) × (number of servings chosen):  
    Minimize:  sum over i of  Cost[i] * x[i]
7.  **Formulate Constraints:**
    -   Constraint 1 (Calories minimum):  
        sum over i of  Calories[i] * x[i]  ≥ 2000
    -   Constraint 2 (Protein minimum):  
        sum over i of  Protein(g)[i] * x[i]  ≥ 50
    -   Constraint 3 (Vitamin C minimum):  
        sum over i of  VitaminC(mg)[i] * x[i]  ≥ 60
    -   Constraint 4 (Fat maximum):  
        sum over i of  Fat(g)[i] * x[i]  ≤ 70
    -   Constraint 5 (Non-negativity):  
        For all i, x[i] ≥ 0
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum for fat. All per-serving nutrient and cost data are provided in the `cost.csv` file.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is the set of Foods, denoted as `i ∈ Foods`, where each food is a row in `cost.csv`.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings (can be fractional) of food `i` to include in the meal plan. Type: GRB.CONTINUOUS, with lower bound 0.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: `Cost` column in `cost.csv` gives the cost per serving for each food.
    -   Constraint coefficients:
        -   `Calories` column: calories per serving of food `i`.
        -   `Protein(g)` column: protein (g) per serving of food `i`.
        -   `Fat(g)` column: fat (g) per serving of food `i`.
        -   `VitaminC(mg)` column: vitamin C (mg) per serving of food `i`.
    -   Constraint RHS (limits):
        -   Calories: at least 2000 kcal.
        -   Protein: at least 50 g.
        -   Vitamin C: at least 60 mg.
        -   Fat: no more than 70 g.
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize the sum over all foods of (cost per serving) × (number of servings):  
    `minimize sum_{i ∈ Foods} Cost[i] * x[i]`
7.  **Formulate Constraints:**
    -   Calorie requirement:  
        `sum_{i ∈ Foods} Calories[i] * x[i] ≥ 2000`
    -   Protein requirement:  
        `sum_{i ∈ Foods} Protein(g)[i] * x[i] ≥ 50`
    -   Vitamin C requirement:  
        `sum_{i ∈ Foods} VitaminC(mg)[i] * x[i] ≥ 60`
    -   Fat limit:  
        `sum_{i ∈ Foods} Fat(g)[i] * x[i] ≤ 70`
    -   Non-negativity:  
        `x[i] ≥ 0` for all `i ∈ Foods`
[Abstract Model Plan END]
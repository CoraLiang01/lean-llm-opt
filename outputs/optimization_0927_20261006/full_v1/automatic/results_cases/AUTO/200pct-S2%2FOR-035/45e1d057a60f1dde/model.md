[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while ensuring the meal plan meets minimum requirements for calories, protein, and vitamin C, and does not exceed a maximum fat limit. All relevant nutritional and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is the set of Foods, denoted as `i ∈ Foods`, where Foods are all rows in the cost.csv file.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of servings (can be fractional) of food `i` to include in the meal plan. Type: GRB.CONTINUOUS, with `x[i] ≥ 0`.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: `Cost` column (cost per serving of food `i`).
    -   Constraint coefficients:
        -   Calories per serving: `Calories` column.
        -   Protein per serving: `Protein(g)` column.
        -   Fat per serving: `Fat(g)` column.
        -   Vitamin C per serving: `VitaminC(mg)` column.
    -   Constraint RHS (limits):
        -   Calories: minimum 2000.
        -   Protein: minimum 50 g.
        -   Vitamin C: minimum 60 mg.
        -   Fat: maximum 70 g.
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize the sum over all foods of (`Cost[i]` × `x[i]`).
7.  **Formulate Constraints:**
    -   Calorie constraint: The sum over all foods of (`Calories[i]` × `x[i]`) ≥ 2000.
    -   Protein constraint: The sum over all foods of (`Protein(g)[i]` × `x[i]`) ≥ 50.
    -   Vitamin C constraint: The sum over all foods of (`VitaminC(mg)[i]` × `x[i]`) ≥ 60.
    -   Fat constraint: The sum over all foods of (`Fat(g)[i]` × `x[i]`) ≤ 70.
    -   Non-negativity: For all foods `i`, `x[i]` ≥ 0.
[Abstract Model Plan END]
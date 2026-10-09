[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum for fat. All nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending/diet problem.
3.  **Define Index Sets:** The primary index is the set of Foods, denoted as $i \in \text{Foods}$ (from all rows in cost.csv).
4.  **Define Decision Variables:**
    -   $x_i$ = Number of servings of food $i$ to include in the meal plan. Type: GRB.CONTINUOUS (non-negative, can be fractional).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving for each food).
    -   Constraint coefficients:
        -   'Calories' (kcal per serving)
        -   'Protein(g)' (grams per serving)
        -   'Fat(g)' (grams per serving)
        -   'VitaminC(mg)' (mg per serving)
    -   Constraint RHS (limits):
        -   Calories: at least 2000
        -   Protein: at least 50 g
        -   Vitamin C: at least 60 mg
        -   Fat: no more than 70 g
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize $\sum_{i \in \text{Foods}} \text{Cost}_i \cdot x_i$.
7.  **Formulate Constraints:**
    -   Calorie requirement: $\sum_{i \in \text{Foods}} \text{Calories}_i \cdot x_i \geq 2000$
    -   Protein requirement: $\sum_{i \in \text{Foods}} \text{Protein(g)}_i \cdot x_i \geq 50$
    -   Vitamin C requirement: $\sum_{i \in \text{Foods}} \text{VitaminC(mg)}_i \cdot x_i \geq 60$
    -   Fat limit: $\sum_{i \in \text{Foods}} \text{Fat(g)}_i \cdot x_i \leq 70$
    -   Non-negativity: $x_i \geq 0$ for all $i \in \text{Foods}$
[Abstract Model Plan END]
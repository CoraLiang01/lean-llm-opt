[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while ensuring the meal plan meets minimum requirements for calories, protein, and vitamin C, and does not exceed a maximum fat limit. All nutrient and cost data per serving are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is the set of Foods, denoted as $i \in \text{Foods}$, where each food is a row in the CSV.
4.  **Define Decision Variables:**
    -   $x_i$ = Number of servings (can be fractional) of food $i$ to include in the meal plan. Type: GRB.CONTINUOUS, $x_i \geq 0$.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column gives the cost per serving for each food $i$.
    -   Constraint coefficients: 'Calories', 'Protein(g)', 'Fat(g)', and 'VitaminC(mg)' columns give the per-serving nutrient content for each food $i$.
    -   Constraint RHS (limits): Minimum total Calories (2000), minimum total Protein (50 g), minimum total Vitamin C (60 mg), and maximum total Fat (70 g).
6.  **Formulate Objective:** Minimize the total cost of the meal plan, i.e., minimize $\sum_{i \in \text{Foods}} \text{Cost}_i \cdot x_i$.
7.  **Formulate Constraints:**
    -   Calorie constraint: $\sum_{i \in \text{Foods}} \text{Calories}_i \cdot x_i \geq 2000$
    -   Protein constraint: $\sum_{i \in \text{Foods}} \text{Protein(g)}_i \cdot x_i \geq 50$
    -   Vitamin C constraint: $\sum_{i \in \text{Foods}} \text{VitaminC(mg)}_i \cdot x_i \geq 60$
    -   Fat constraint: $\sum_{i \in \text{Foods}} \text{Fat(g)}_i \cdot x_i \leq 70$
    -   Non-negativity: $x_i \geq 0$ for all $i \in \text{Foods}$
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to design a one-day meal plan by selecting servings (possibly fractional) of available foods to minimize total cost, while meeting minimum requirements for calories, protein, and vitamin C, and not exceeding a maximum fat limit. All per-serving nutrient and cost data are provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) blending problem.
3.  **Define Index Sets:** The primary index is the set of Foods, denoted as $i \in \text{Foods}$ (all rows in the 'Food' column of cost.csv).
4.  **Define Decision Variables:**
    -   $x_i$ = number of servings of food $i$ to include in the meal plan. Type: GRB.CONTINUOUS (non-negative, can be fractional).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Cost' column (cost per serving for each food).
    -   Constraint coefficients:
        -   'Calories' (calories per serving)
        -   'Protein(g)' (protein per serving)
        -   'VitaminC(mg)' (vitamin C per serving)
        -   'Fat(g)' (fat per serving)
    -   Constraint RHS (limits): Provided in the query (Calories ≥ 2000, Protein ≥ 50g, Vitamin C ≥ 60mg, Fat ≤ 70g).
6.  **Formulate Objective:** Minimize the total cost of the meal plan: $\min \sum_{i \in \text{Foods}} \text{Cost}_i \cdot x_i$.
7.  **Formulate Constraints:**
    -   Calorie requirement: $\sum_{i \in \text{Foods}} \text{Calories}_i \cdot x_i \geq 2000$
    -   Protein requirement: $\sum_{i \in \text{Foods}} \text{Protein(g)}_i \cdot x_i \geq 50$
    -   Vitamin C requirement: $\sum_{i \in \text{Foods}} \text{VitaminC(mg)}_i \cdot x_i \geq 60$
    -   Fat limit: $\sum_{i \in \text{Foods}} \text{Fat(g)}_i \cdot x_i \leq 70$
    -   Non-negativity: $x_i \geq 0$ for all $i \in \text{Foods}$
[Abstract Model Plan END]
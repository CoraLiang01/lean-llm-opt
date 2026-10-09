[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), subject to blending proportion requirements for each brand, raw material daily supply limits, and a minimum production requirement for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a blending linear programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Raw Grades: \( G = \{\text{I}, \text{II}, \text{III}\} \)
    - Brands: \( B = \{\text{Red}, \text{Yellow}, \text{Blue}\} \)
4.  **Define Decision Variables:**
    - \( x_{g,b} \) = amount (kg) of raw grade \( g \) used in brand \( b \). Type: GRB.CONTINUOUS (non-negative).
5.  **Identify Parameters (from Schema):**
    - From 30-1.csv:
        - \( \text{Supply}_g \): 'Daily Supply (kg)' for each grade \( g \)
        - \( \text{Cost}_g \): 'Cost (CNY/kg)' for each grade \( g \)
    - From 30-2.csv:
        - \( \text{Price}_b \): 'Selling Price (CNY/kg)' for each brand \( b \)
        - Blending requirements for each brand \( b \): upper/lower bounds on the proportion of certain grades in the blend
6.  **Formulate Objective:** Maximize total net profit:
    - \( \sum_{b \in B} \text{Price}_b \cdot \left(\sum_{g \in G} x_{g,b}\right) - \sum_{g \in G} \text{Cost}_g \cdot \left(\sum_{b \in B} x_{g,b}\right) \)
7.  **Formulate Constraints:**
    - Constraint 1 (Blending Requirements): For each brand \( b \), enforce the specified upper/lower bounds on the proportion of selected grades in the blend:
        - For each required grade \( g \) in brand \( b \): \( \text{LowerBound}_{g,b} \leq \frac{x_{g,b}}{\sum_{g' \in G} x_{g',b}} \leq \text{UpperBound}_{g,b} \), if specified.
    - Constraint 2 (Raw Material Supply): For each grade \( g \), \( \sum_{b \in B} x_{g,b} \leq \text{Supply}_g \).
    - Constraint 3 (Minimum Production for Red): \( \sum_{g \in G} x_{g,\text{Red}} \geq 2000 \).
    - Constraint 4 (Non-negativity): \( x_{g,b} \geq 0 \) for all \( g \in G, b \in B \).
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), while satisfying blending proportion requirements, raw material supply limits, and a minimum production level for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a blending linear programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material (from 30-1.csv): \( G \) (e.g., I, II, III)
    - Wine brands (from 30-2.csv): \( B \) (e.g., Red, Yellow, Blue)
4.  **Define Decision Variables:**
    -   \( x_{g,b} \) = quantity (kg) of raw grade \( g \) allocated to brand \( b \). Type: GRB.CONTINUOUS (non-negative real).
5.  **Identify Parameters (from Schema):**
    -   Selling prices per brand: from 'Selling Price (CNY/kg)' in 30-2.csv.
    -   Raw material costs per grade: from 'Cost (CNY/kg)' in 30-1.csv.
    -   Raw material daily supply limits: from 'Daily Supply (kg)' in 30-1.csv.
    -   Blending requirements (upper/lower bounds on proportions of certain grades in each brand): from 'Blending Requirements' in 30-2.csv.
    -   Minimum production for Red brand: fixed at 2,000 kg.
6.  **Formulate Objective:** Maximize total net profit, calculated as:
        - Total sales revenue: sum over brands of (selling price per kg) × (total kg produced of each brand)
        - Minus total raw material cost: sum over all grades and brands of (cost per kg of grade) × (kg of grade used in each brand)
        - Symbolically: maximize \( \sum_{b \in B} \text{Price}_b \cdot \sum_{g \in G} x_{g,b} - \sum_{g \in G} \text{Cost}_g \cdot \sum_{b \in B} x_{g,b} \)
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand and each specified grade, enforce upper and/or lower bounds on the proportion of that grade in the brand's blend. For example, for brand \( b \) and grade \( g \), if the requirement is "grade \( g \) less than \( U\% \)", then \( x_{g,b} \leq U\% \cdot \sum_{g' \in G} x_{g',b} \); if "grade \( g \) more than \( L\% \)", then \( x_{g,b} \geq L\% \cdot \sum_{g' \in G} x_{g',b} \).
    -   Constraint 2 (Raw Material Supply): For each grade \( g \), the total amount used across all brands cannot exceed its daily supply limit: \( \sum_{b \in B} x_{g,b} \leq \text{Supply}_g \).
    -   Constraint 3 (Minimum Production for Red): The total daily production of the Red brand must be at least 2,000 kg: \( \sum_{g \in G} x_{g,\text{Red}} \geq 2000 \).
    -   Constraint 4 (Non-negativity): All \( x_{g,b} \geq 0 \).
[Abstract Model Plan END]
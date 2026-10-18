[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), while satisfying blending proportion requirements for each brand, raw material supply limits, and a minimum production level for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Blending Linear Programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material (from 30-1.csv): \( G \)
    - Wine brands (from 30-2.csv): \( B \)
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade \( g \) used in brand \( b \). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    -   Raw material supply limits: 'Daily Supply (kg)' from 30-1.csv, indexed by grade \( g \).
    -   Raw material costs: 'Cost (CNY/kg)' from 30-1.csv, indexed by grade \( g \).
    -   Selling prices: 'Selling Price (CNY/kg)' from 30-2.csv, indexed by brand \( b \).
    -   Blending requirements: 'Blending Requirements' from 30-2.csv, parsed into lower and upper bounds on the proportion of certain grades in each brand.
6.  **Formulate Objective:** Maximize total net profit, calculated as the sum over all brands of (total sales revenue for each brand minus total raw material cost for that brand):  
    \[
    \text{Maximize} \quad \sum_{b \in B} \left[ \text{Selling Price}_b \cdot \left( \sum_{g \in G} x[g, b] \right) - \sum_{g \in G} \text{Cost}_g \cdot x[g, b] \right]
    \]
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand \( b \) and each grade \( g \) with a specified lower or upper bound, enforce:
        - Lower bound: \( x[g, b] \geq \text{LowerBound}_{g, b} \times \sum_{g' \in G} x[g', b] \)
        - Upper bound: \( x[g, b] \leq \text{UpperBound}_{g, b} \times \sum_{g' \in G} x[g', b] \)
    -   Constraint 2 (Raw Material Supply): For each grade \( g \), the total used across all brands cannot exceed its daily supply:
        \[
        \sum_{b \in B} x[g, b] \leq \text{Daily Supply}_g
        \]
    -   Constraint 3 (Minimum Production for Red): The total daily production of the Red brand must be at least 2,000 kg:
        \[
        \sum_{g \in G} x[g, \text{Red}] \geq 2000
        \]
    -   Constraint 4 (Non-negativity): All decision variables must be non-negative:
        \[
        x[g, b] \geq 0 \quad \forall g \in G, b \in B
        \]
[Abstract Model Plan END]
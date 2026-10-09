[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), subject to blending proportion requirements, raw material supply limits, and a minimum production requirement for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a blending linear programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Raw Grades: \( G = \{\text{I}, \text{II}, \text{III}\} \)
    - Brands: \( B = \{\text{Red}, \text{Yellow}, \text{Blue}\} \)
4.  **Define Decision Variables:**
    -   \( x_{g,b} \) = amount (kg) of raw grade \( g \) used in brand \( b \). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    -   Raw grade daily supply limits: from 30-1.csv, column 'Daily Supply (kg)'.
    -   Raw grade unit costs: from 30-1.csv, column 'Cost (CNY/kg)'.
    -   Brand selling prices: from 30-2.csv, column 'Selling Price (CNY/kg)'.
    -   Blending requirements: from 30-2.csv, column 'Blending Requirements' (upper/lower bounds on proportions of certain grades in each brand).
6.  **Formulate Objective:** Maximize total net profit, i.e., maximize sum over brands of (total sales revenue per brand) minus sum over all grades and brands of (raw material cost):
    - Maximize \( \sum_{b \in B} \text{SellingPrice}_b \cdot \left(\sum_{g \in G} x_{g,b}\right) - \sum_{g \in G} \text{Cost}_g \cdot \left(\sum_{b \in B} x_{g,b}\right) \).
7.  **Formulate Constraints:**
    -   Constraint 1 (Raw Material Supply): For each grade \( g \), \( \sum_{b \in B} x_{g,b} \leq \text{DailySupply}_g \).
    -   Constraint 2 (Blending Requirements): For each brand \( b \), enforce the specified upper and/or lower bounds on the proportion of selected grades in the blend, i.e., for each requirement "grade \( g \) less than \( p\% \)" or "grade \( g \) more than \( p\% \)", require \( x_{g,b} \leq p/100 \cdot \sum_{g' \in G} x_{g',b} \) or \( x_{g,b} \geq p/100 \cdot \sum_{g' \in G} x_{g',b} \), respectively.
    -   Constraint 3 (Minimum Production for Red): \( \sum_{g \in G} x_{g,\text{Red}} \geq 2000 \).
    -   Constraint 4 (Non-negativity): \( x_{g,b} \geq 0 \) for all \( g \in G, b \in B \).
[Abstract Model Plan END]
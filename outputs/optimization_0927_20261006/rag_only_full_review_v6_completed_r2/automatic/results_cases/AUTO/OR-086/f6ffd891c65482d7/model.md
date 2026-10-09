[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), subject to blending proportion requirements, raw material supply limits, and a minimum production requirement for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a blending linear programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    -   Grades of raw wine material (from 30-1.csv): \( G \)
    -   Wine brands (from 30-2.csv): \( B \)
4.  **Define Decision Variables:**
    -   \( x_{g,b} \) = quantity (kg) of raw grade \( g \) allocated to brand \( b \). Type: GRB.CONTINUOUS, \( x_{g,b} \geq 0 \).
5.  **Identify Parameters (from Schema):**
    -   From 30-1.csv:
        -   \( \text{Supply}_g \): 'Daily Supply (kg)' for grade \( g \)
        -   \( \text{Cost}_g \): 'Cost (CNY/kg)' for grade \( g \)
    -   From 30-2.csv:
        -   \( \text{Price}_b \): 'Selling Price (CNY/kg)' for brand \( b \)
        -   Blending requirements for each brand \( b \): lower and upper bounds on the proportion of certain grades in the blend (parsed from 'Blending Requirements')
    -   Minimum production for Red brand: 2,000 kg (from query)
6.  **Formulate Objective:** Maximize total net profit, calculated as:
    -   Total sales revenue: sum over all brands of (total kg produced of brand \( b \)) × (selling price of brand \( b \))
    -   Minus total raw material cost: sum over all grades and brands of (kg of grade \( g \) used in brand \( b \)) × (cost per kg of grade \( g \))
    -   Objective: Maximize \( \sum_{b \in B} \left( \text{Price}_b \cdot \sum_{g \in G} x_{g,b} \right) - \sum_{g \in G} \sum_{b \in B} \left( \text{Cost}_g \cdot x_{g,b} \right) \)
7.  **Formulate Constraints:**
    -   **Blending Requirements:** For each brand \( b \) and each grade \( g \) with a specified lower or upper proportion bound, enforce:
        -   Lower bound: \( \frac{x_{g,b}}{\sum_{g' \in G} x_{g',b}} \geq \text{LowerBound}_{g,b} \) (if specified)
        -   Upper bound: \( \frac{x_{g,b}}{\sum_{g' \in G} x_{g',b}} \leq \text{UpperBound}_{g,b} \) (if specified)
        -   These are implemented as linear constraints by cross-multiplying denominators.
    -   **Raw Material Supply Constraints:** For each grade \( g \), the total amount used across all brands cannot exceed its daily supply:
        -   \( \sum_{b \in B} x_{g,b} \leq \text{Supply}_g \)
    -   **Minimum Production for Red Brand:** The total production of the Red brand must be at least 2,000 kg:
        -   \( \sum_{g \in G} x_{g,\text{Red}} \geq 2000 \)
    -   **Non-negativity:** All \( x_{g,b} \geq 0 \)
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), while satisfying blending proportion requirements for each brand, raw material supply limits, and a minimum production level for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a blending linear programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material (from 30-1.csv): \( G = \{\text{I}, \text{II}, \text{III}\} \)
    - Wine brands (from 30-2.csv): \( B = \{\text{Red}, \text{Yellow}, \text{Blue}\} \)
4.  **Define Decision Variables:**
    -   \( x_{g,b} \) = amount (kg) of raw grade \( g \) used in brand \( b \). Type: GRB.CONTINUOUS (non-negative real).
5.  **Identify Parameters (from Schema):**
    -   Raw material supply limits: 'Daily Supply (kg)' from 30-1.csv, indexed by grade \( g \).
    -   Raw material costs: 'Cost (CNY/kg)' from 30-1.csv, indexed by grade \( g \).
    -   Selling prices: 'Selling Price (CNY/kg)' from 30-2.csv, indexed by brand \( b \).
    -   Blending requirements: 'Blending Requirements' from 30-2.csv, specifying upper and lower bounds on the proportion of certain grades in each brand.
    -   Minimum production for Red: fixed lower bound (2,000 kg) on total production of Red brand.
6.  **Formulate Objective:** Maximize total net profit, calculated as:
        - Total sales revenue: sum over all brands of (selling price per kg) × (total kg produced of that brand)
        - Minus total raw material cost: sum over all grades and brands of (cost per kg) × (kg of grade used in brand)
        - Symbolically: maximize \( \sum_{b \in B} \text{Price}_b \cdot \sum_{g \in G} x_{g,b} - \sum_{g \in G} \text{Cost}_g \cdot \sum_{b \in B} x_{g,b} \)
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand \( b \) and each specified grade \( g \), enforce lower and/or upper bounds on the proportion of grade \( g \) in the total blend for brand \( b \), as specified in 'Blending Requirements'. For example, if brand Red requires grade I less than 10%, then \( x_{\text{I},\text{Red}} / \sum_{g'} x_{g',\text{Red}} < 0.10 \); if grade II more than 50%, then \( x_{\text{II},\text{Red}} / \sum_{g'} x_{g',\text{Red}} > 0.50 \), and similarly for other brands and grades.
    -   Constraint 2 (Raw Material Supply): For each grade \( g \), the total amount used across all brands cannot exceed its daily supply limit: \( \sum_{b \in B} x_{g,b} \leq \text{Daily Supply}_g \).
    -   Constraint 3 (Minimum Production for Red): The total production of the Red brand must be at least 2,000 kg: \( \sum_{g \in G} x_{g,\text{Red}} \geq 2000 \).
    -   Constraint 4 (Non-negativity): All \( x_{g,b} \geq 0 \).
[Abstract Model Plan END]
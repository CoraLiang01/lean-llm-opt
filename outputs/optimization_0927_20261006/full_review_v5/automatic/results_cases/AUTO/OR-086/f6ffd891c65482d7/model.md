[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), subject to blending proportion requirements, raw material supply limits, and a minimum production level for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a blending linear programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material (from 30-1.csv): \( G \)
    - Wine brands (from 30-2.csv): \( B \)
4.  **Define Decision Variables:**
    -   \( x_{g,b} \) = quantity (kg) of raw grade \( g \) allocated to brand \( b \). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    -   Selling prices per brand: from 30-2.csv, column 'Selling Price (CNY/kg)'.
    -   Raw material costs per grade: from 30-1.csv, column 'Cost (CNY/kg)'.
    -   Raw material daily supply limits: from 30-1.csv, column 'Daily Supply (kg)'.
    -   Blending requirements (upper/lower bounds on proportions of certain grades in each brand): from 30-2.csv, column 'Blending Requirements'.
    -   Minimum production for Red brand: fixed at 2,000 kg (from query).
6.  **Formulate Objective:** Maximize total net profit, i.e., maximize sum over brands of (total sales revenue per brand) minus sum over grades of (total raw material cost per grade):
    -   Objective: maximize \( \sum_{b \in B} \text{SellingPrice}_b \cdot \left(\sum_{g \in G} x_{g,b}\right) - \sum_{g \in G} \text{Cost}_g \cdot \left(\sum_{b \in B} x_{g,b}\right) \).
7.  **Formulate Constraints:**
    -   Constraint 1 (Blending Requirements): For each brand \( b \) and each specified grade \( g \), enforce lower and/or upper bounds on the proportion of grade \( g \) in the total blend for brand \( b \), as specified in 'Blending Requirements' (e.g., \( x_{g,b} / \sum_{g' \in G} x_{g',b} \leq \text{upper bound} \), \( x_{g,b} / \sum_{g' \in G} x_{g',b} \geq \text{lower bound} \)), whenever the total production for brand \( b \) is positive.
    -   Constraint 2 (Raw Material Supply): For each grade \( g \), \( \sum_{b \in B} x_{g,b} \leq \) 'Daily Supply (kg)' for grade \( g \).
    -   Constraint 3 (Minimum Production for Red): \( \sum_{g \in G} x_{g,\text{Red}} \geq 2,000 \) kg.
    -   Constraint 4 (Non-negativity): \( x_{g,b} \geq 0 \) for all \( g \in G, b \in B \).
[Abstract Model Plan END]
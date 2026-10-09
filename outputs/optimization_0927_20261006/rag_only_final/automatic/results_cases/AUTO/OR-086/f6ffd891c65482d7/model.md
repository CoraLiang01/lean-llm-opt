[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), subject to blending proportion requirements, raw material supply limits, and a minimum production level for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a blending linear programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material (from 30-1.csv): \( G \)
    - Wine brands (from 30-2.csv): \( B \)
4.  **Define Decision Variables:**
    -   \( x_{g,b} \) = quantity (kg) of raw grade \( g \) allocated to brand \( b \). Type: GRB.CONTINUOUS (non-negative real).
5.  **Identify Parameters (from Schema):**
    -   Selling prices per brand: from 30-2.csv, column 'Selling Price (CNY/kg)' (indexed by \( b \)).
    -   Raw material costs per grade: from 30-1.csv, column 'Cost (CNY/kg)' (indexed by \( g \)).
    -   Daily supply limits per grade: from 30-1.csv, column 'Daily Supply (kg)' (indexed by \( g \)).
    -   Blending requirements: from 30-2.csv, column 'Blending Requirements' (parsed for lower/upper bounds on proportions of grades in each brand).
    -   Minimum production for Red brand: fixed at 2,000 kg.
6.  **Formulate Objective:** Maximize total net profit, calculated as:
    -   Total sales revenue: sum over brands of (selling price per kg) × (total kg produced of each brand).
    -   Minus total raw material cost: sum over all grades and brands of (cost per kg) × (kg of grade used in brand).
    -   Symbolically: Maximize \( \sum_{b \in B} \text{Price}_b \cdot \left(\sum_{g \in G} x_{g,b}\right) - \sum_{g \in G} \sum_{b \in B} \text{Cost}_g \cdot x_{g,b} \).
7.  **Formulate Constraints:**
    -   Blending Requirements: For each brand \( b \) and each specified grade \( g \), enforce lower and/or upper bounds on the proportion of grade \( g \) in the total production of brand \( b \), i.e., \( \text{Lower}_{g,b} \leq \frac{x_{g,b}}{\sum_{g' \in G} x_{g',b}} \leq \text{Upper}_{g,b} \), where bounds are parsed from 'Blending Requirements'.
    -   Raw Material Supply Limits: For each grade \( g \), the total amount used across all brands cannot exceed its daily supply, i.e., \( \sum_{b \in B} x_{g,b} \leq \text{Supply}_g \).
    -   Minimum Production for Red Brand: The total production of the Red brand must be at least 2,000 kg, i.e., \( \sum_{g \in G} x_{g,\text{Red}} \geq 2000 \).
    -   Non-negativity: All \( x_{g,b} \geq 0 \).
[Abstract Model Plan END]
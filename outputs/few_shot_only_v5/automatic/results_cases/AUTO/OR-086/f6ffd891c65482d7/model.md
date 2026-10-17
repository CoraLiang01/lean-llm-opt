[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three raw wine grades to three wine brands, maximizing total net profit (sales revenue minus raw material cost), subject to blending proportion requirements, raw material supply limits, and a minimum production level for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a blending linear programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Grades of raw wine material: \( G = \{\text{I}, \text{II}, \text{III}\} \)
    - Wine brands: \( B = \{\text{Red}, \text{Yellow}, \text{Blue}\} \)
4.  **Define Decision Variables:**
    - \( x_{g,b} \) = amount (kg) of raw grade \( g \) used in brand \( b \). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    - From 30-1.csv:
        - \( \text{Supply}_g \): 'Daily Supply (kg)' for each grade \( g \)
        - \( \text{Cost}_g \): 'Cost (CNY/kg)' for each grade \( g \)
    - From 30-2.csv:
        - \( \text{Price}_b \): 'Selling Price (CNY/kg)' for each brand \( b \)
        - Blending requirements for each brand \( b \): upper/lower bounds on the proportion of certain grades in the blend
6.  **Formulate Objective:** Maximize total net profit, i.e., maximize
    \[
    \sum_{b \in B} \text{Price}_b \cdot \left(\sum_{g \in G} x_{g,b}\right) - \sum_{g \in G} \text{Cost}_g \cdot \left(\sum_{b \in B} x_{g,b}\right)
    \]
7.  **Formulate Constraints:**
    -   **Blending Requirements (for each brand):**
        - For each brand \( b \), for each specified grade \( g \), enforce the stated upper or lower bound on the proportion of \( g \) in the total blend for \( b \):
            - For Red: \( \frac{x_{\text{I},\text{Red}}}{\sum_{g} x_{g,\text{Red}}} < 0.10 \), \( \frac{x_{\text{II},\text{Red}}}{\sum_{g} x_{g,\text{Red}}} > 0.50 \)
            - For Yellow: \( \frac{x_{\text{III},\text{Yellow}}}{\sum_{g} x_{g,\text{Yellow}}} < 0.70 \), \( \frac{x_{\text{I},\text{Yellow}}}{\sum_{g} x_{g,\text{Yellow}}} > 0.20 \)
            - For Blue: \( \frac{x_{\text{I},\text{Blue}}}{\sum_{g} x_{g,\text{Blue}}} < 0.50 \), \( \frac{x_{\text{II},\text{Blue}}}{\sum_{g} x_{g,\text{Blue}}} > 0.10 \)
        - For each brand, if the denominator is zero (no production), the constraint is vacuously satisfied.
    -   **Raw Material Supply Constraints:** For each grade \( g \), the total amount used across all brands cannot exceed its daily supply:
        \[
        \sum_{b \in B} x_{g,b} \leq \text{Supply}_g
        \]
    -   **Minimum Production for Red:** The total daily production of the Red brand must be at least 2,000 kg:
        \[
        \sum_{g \in G} x_{g,\text{Red}} \geq 2000
        \]
    -   **Non-negativity:** All \( x_{g,b} \geq 0 \).
[Abstract Model Plan END]
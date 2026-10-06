[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal daily allocation of three grades of raw wine to three wine brands, maximizing total net profit (sales revenue minus raw material cost), subject to blending requirements, raw material supply limits, and a minimum production level for the Red brand.
2.  **Identify Model Type:** Based on the query, this is a Blending Linear Programming (LP) problem.
3.  **Define Index Sets:** The primary indices are:
    -   Grades of raw wine material (from 30-1.csv): {I, II, III}
    -   Wine brands (from 30-2.csv): {Red, Yellow, Blue}
4.  **Define Decision Variables:**
    -   `x[g, b]` = Amount (kg) of raw grade `g` used in brand `b`. Type: GRB.CONTINUOUS (non-negative real numbers).
5.  **Identify Parameters (from Schema):**
    -   From 30-1.csv:
        -   `Supply[g]`: Daily supply limit for grade `g` ('Daily Supply (kg)')
        -   `Cost[g]`: Unit cost for grade `g` ('Cost (CNY/kg)')
    -   From 30-2.csv:
        -   `Price[b]`: Selling price per kg for brand `b` ('Selling Price (CNY/kg)')
        -   `BlendingReq[b]`: Blending requirements for brand `b` ('Blending Requirements'), specifying upper/lower bounds on the proportion of certain grades in each brand.
6.  **Formulate Objective:** Maximize total net profit, calculated as:
    -   Total sales revenue: sum over all brands of (total kg produced of brand `b`) × (selling price of brand `b`)
    -   Minus total raw material cost: sum over all grades and brands of (kg of grade `g` used in brand `b`) × (cost of grade `g`)
    -   In formula: Maximize  
        sum_b [ (sum_g x[g, b]) × Price[b] ] − sum_g sum_b [ x[g, b] × Cost[g] ]
7.  **Formulate Constraints:**
    -   **Blending Requirements:** For each brand and each specified grade in its blending requirements, enforce:
        -   Lower bound: (x[g, b]) / (sum_g' x[g', b]) ≥ required minimum proportion (if specified)
        -   Upper bound: (x[g, b]) / (sum_g' x[g', b]) ≤ required maximum proportion (if specified)
        -   (The specific bounds are parsed from the 'Blending Requirements' column for each brand.)
    -   **Raw Material Supply Constraints:** For each grade `g`, the total amount used across all brands cannot exceed its daily supply limit:
        -   sum_b x[g, b] ≤ Supply[g]
    -   **Minimum Production for Red Brand:** The total daily production of the Red brand must be at least 2,000 kg:
        -   sum_g x[g, 'Red'] ≥ 2,000
    -   **Non-negativity:** All decision variables must be non-negative:
        -   x[g, b] ≥ 0 for all grades and brands
[Abstract Model Plan END]
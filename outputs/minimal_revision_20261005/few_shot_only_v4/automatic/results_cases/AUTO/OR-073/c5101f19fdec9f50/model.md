[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal continuous production plan for three products (I, II, III), each requiring two sequential procedures (A and B), using a set of specified equipment for each procedure and product, to maximize profit. The model must account for processing times, raw material costs, selling prices, available equipment operating times, and equipment costs at full load, as provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) production planning and assignment problem with assignment restrictions.
3.  **Define Index Sets:** The primary indices are:
    - Products: {I, II, III}
    - Procedures: {A, B}
    - Equipment: {A1, A2, B1, B2, B3} (filtered per product and procedure as described in the query)
4.  **Define Decision Variables:**
    -   `x[p]` = Quantity of product p produced (p ∈ {I, II, III}). Type: GRB.CONTINUOUS.
    -   `z[p,e]` = Amount of product p processed on equipment e for its respective procedure (only for allowed (p,e) pairs). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        - Selling price per unit: from row 'Unit Price (yuan/unit)' for each product.
        - Raw material cost per unit: from row 'Raw Material Cost (yuan/unit)' for each product.
        - Equipment cost at full load: from column 'Equipment Cost at Full Load (yuan)' for each equipment.
    -   Constraint coefficients:
        - Processing time per unit: from the intersection of equipment row and product column (e.g., A1/Product I).
        - Available equipment operating time: from column 'Available Equipment Operating Time' for each equipment.
    -   Constraint RHS:
        - Equipment operating time limits: from 'Available Equipment Operating Time' for each equipment.
6.  **Formulate Objective:** Maximize total profit, defined as:
        - Total revenue from all products (sum over products of unit price × quantity produced)
        - Minus total raw material costs (sum over products of raw material cost × quantity produced)
        - Minus total equipment costs (sum over all equipment of (actual usage fraction × equipment cost at full load), where usage fraction = total time used / available time for each equipment)
7.  **Formulate Constraints:**
    -   Constraint 1 (Production Assignment): For each product and each procedure, the total quantity produced must be assigned to the allowed equipment for that procedure:
        - For each product p and procedure s, sum over allowed equipment e of z[p,e] = x[p].
    -   Constraint 2 (Equipment Time Limits): For each equipment e, the total processing time assigned to it across all products must not exceed its available operating time:
        - For each equipment e, sum over products p of (processing time per unit for (p,e)) × z[p,e] ≤ available operating time for e.
    -   Constraint 3 (Equipment-Product Assignment Restrictions): Only allow z[p,e] > 0 for equipment-product pairs permitted by the query:
        - Product I: Procedure A on A1 or A2; Procedure B on B1, B2, or B3.
        - Product II: Procedure A on A1 or A2; Procedure B only on B1.
        - Product III: Procedure A only on A2; Procedure B only on B2.
    -   Constraint 4 (Non-negativity): All variables x[p] and z[p,e] ≥ 0.
[Abstract Model Plan END]
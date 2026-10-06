[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal continuous production plan for three products (I, II, III), each requiring two sequential procedures (A and B), using a set of specified equipment for each procedure and product, to maximize profit. The model must account for processing times, raw material costs, selling prices, available equipment operating times, and equipment costs at full load, as provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) production planning and assignment problem with assignment restrictions.
3.  **Define Index Sets:** The primary indices are:
    - Products: {I, II, III}
    - Procedures: {A, B}
    - Equipment: {A1, A2, B1, B2, B3} (filtered per product and procedure as described in the query)
4.  **Define Decision Variables:**
    -   `x[p]` = Quantity of product p produced (continuous, for p in {I, II, III}). Type: GRB.CONTINUOUS.
    -   `z[p,e]` = Amount of product p processed on equipment e for its respective procedure (continuous, only for allowed (p,e) pairs). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    -   Selling price per unit: from row 'Unit Price (yuan/unit)', columns 'Product I', 'Product II', 'Product III'.
    -   Raw material cost per unit: from row 'Raw Material Cost (yuan/unit)', columns 'Product I', 'Product II', 'Product III'.
    -   Processing time per unit: from each equipment row, columns 'Product I', 'Product II', 'Product III'.
    -   Available equipment operating time: from each equipment row, column 'Available Equipment Operating Time'.
    -   Equipment cost at full load: from each equipment row, column 'Equipment Cost at Full Load (yuan)'.
6.  **Formulate Objective:** Maximize total profit, defined as:
    - Total revenue from all products (sum over p: selling price[p] * x[p])
    - Minus total raw material costs (sum over p: raw material cost[p] * x[p])
    - Minus total equipment costs (sum over all equipment e: (total time used on e / available time of e) * equipment cost at full load for e)
7.  **Formulate Constraints:**
    -   Constraint 1 (Production Assignment): For each product p and each procedure (A or B), the total amount of p processed on allowed equipment for that procedure must equal x[p].
        - For procedure A:
            - Product I: sum over e in {A1, A2} of z[I,e] = x[I]
            - Product II: sum over e in {A1, A2} of z[II,e] = x[II]
            - Product III: z[III,A2] = x[III]
        - For procedure B:
            - Product I: sum over e in {B1, B2, B3} of z[I,e] = x[I]
            - Product II: z[II,B1] = x[II]
            - Product III: z[III,B2] = x[III]
    -   Constraint 2 (Equipment Capacity): For each equipment e, the total processing time assigned to e across all products must not exceed its available operating time.
        - For each e: sum over p (z[p,e] * processing time per unit for (p,e)) ≤ available equipment operating time for e
    -   Constraint 3 (Non-negativity): All variables x[p] and z[p,e] ≥ 0
[Abstract Model Plan END]
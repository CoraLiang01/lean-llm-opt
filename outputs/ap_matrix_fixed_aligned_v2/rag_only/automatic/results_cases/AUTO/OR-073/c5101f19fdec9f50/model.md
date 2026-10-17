[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal continuous production quantities of three products (I, II, III) in a factory, where each product must be processed through two procedures (A and B) using specific eligible equipment types. The goal is to maximize profit, considering processing times, raw material costs, selling prices, available equipment operating times, and equipment costs at full load, as provided in the CSV schema.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) production planning problem with assignment and resource constraints.
3.  **Define Index Sets:** The primary indices are:
    - Products: {I, II, III}
    - Equipment: {A1, A2, B1, B2, B3} (filtered per product and procedure eligibility)
    - Procedures: {A, B}
4.  **Define Decision Variables:**
    - `x[p]` = Quantity of product p produced (for p in Products). Type: GRB.CONTINUOUS.
    - `z[p,e]` = Amount of product p processed on equipment e (for eligible (p,e) pairs per procedure). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - Selling price per unit: from 'Unit Price (yuan/unit)' row, columns 'Product I', 'Product II', 'Product III'.
        - Raw material cost per unit: from 'Raw Material Cost (yuan/unit)' row, columns as above.
        - Equipment cost at full load: from 'Equipment Cost at Full Load (yuan)' column, for each equipment.
    - Constraint coefficients:
        - Processing time per unit: from each equipment row, columns 'Product I', 'Product II', 'Product III'.
        - Available equipment operating time: from 'Available Equipment Operating Time' column, for each equipment.
    - Constraint RHS:
        - Equipment operating time limits: from 'Available Equipment Operating Time' column.
6.  **Formulate Objective:** Maximize total profit, defined as:
        Total revenue from all products (sum over p: selling price[p] * x[p])
        minus total raw material costs (sum over p: raw material cost[p] * x[p])
        minus total equipment operating costs (sum over e: equipment cost at full load[e] * (total time used on e / available time[e])).
7.  **Formulate Constraints:**
    - Constraint 1 (Production-Processing Consistency): For each product p and each procedure (A and B), the total amount of product p processed across all eligible equipment for that procedure must equal x[p].
    - Constraint 2 (Equipment Time Limits): For each equipment e, the total processing time used (sum over eligible products p: processing time per unit[p,e] * z[p,e]) must not exceed the available operating time for equipment e.
    - Constraint 3 (Eligibility): Only allow z[p,e] > 0 if product p is eligible to be processed on equipment e for the relevant procedure, as specified in the query (e.g., Product III only on A2 and B2, Product II on B1 for procedure B, etc.).
    - Constraint 4 (Non-negativity): All decision variables x[p] and z[p,e] must be greater than or equal to zero.
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal continuous production plan for three products (I, II, III), each requiring two procedures (A and B), with each procedure performed on specific eligible equipment types. The goal is to maximize profit, considering processing times, raw material costs, selling prices, equipment operating time limits, and equipment costs at full load, as provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) production planning and assignment problem with assignment eligibility constraints.
3.  **Define Index Sets:** The primary indices are:
    - Products: {I, II, III}
    - Procedures: {A, B}
    - Equipment: {A1, A2, B1, B2, B3} (filtered per product and procedure eligibility)
4.  **Define Decision Variables:**
    - `x[p, e]` = Quantity of product `p` processed on equipment `e` for its respective procedure. Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    - Processing time per unit for each (product, equipment) pair: from columns 'Product I', 'Product II', 'Product III' for each equipment row.
    - Raw material cost per unit for each product: from 'Raw Material Cost (yuan/unit)' row.
    - Selling price per unit for each product: from 'Unit Price (yuan/unit)' row.
    - Available equipment operating time: from 'Available Equipment Operating Time' column.
    - Equipment cost at full load: from 'Equipment Cost at Full Load (yuan)' column.
6.  **Formulate Objective:** Maximize total profit, defined as:
        - Total revenue from all products (sum over all products of selling price × total produced units)
        - Minus total raw material costs (sum over all products of raw material cost × total produced units)
        - Minus total equipment costs (sum over all equipment of (operating time used / available time) × equipment cost at full load)
7.  **Formulate Constraints:**
    - Constraint 1 (Production Completion): For each product, the quantity produced must be the same across both procedures (A and B), i.e., sum of `x[p, e]` over eligible A equipment = sum over eligible B equipment, for each product `p`.
    - Constraint 2 (Equipment Assignment Eligibility): Only allow `x[p, e]` > 0 if product `p` is eligible to be processed on equipment `e` for the relevant procedure, as specified in the query.
    - Constraint 3 (Equipment Operating Time): For each equipment, the total processing time used (sum over all assigned products of processing time per unit × quantity assigned) ≤ available equipment operating time.
    - Constraint 4 (Non-negativity): All `x[p, e]` ≥ 0.
[Abstract Model Plan END]
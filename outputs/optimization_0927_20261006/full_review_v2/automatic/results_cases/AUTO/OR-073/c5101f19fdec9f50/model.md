[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal continuous production plan for three products (I, II, III), each requiring two procedures (A and B), with each procedure performed on specific eligible equipment types. The goal is to maximize profit, considering processing times, raw material costs, selling prices, equipment operating time limits, and equipment costs at full load, as provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) production planning and assignment problem with assignment restrictions.
3.  **Define Index Sets:** The primary indices are:
    - Products: {I, II, III}
    - Procedures: {A, B}
    - Equipment: {A1, A2, B1, B2, B3} (filtered per product and procedure eligibility)
4.  **Define Decision Variables:**
    - `x[p, e]` = Quantity of product `p` processed on equipment `e` for its respective procedure. Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    - Processing time per unit: from columns 'Product I', 'Product II', 'Product III' for each equipment row.
    - Raw material cost per unit: from row 'Raw Material Cost (yuan/unit)' for each product.
    - Unit selling price: from row 'Unit Price (yuan/unit)' for each product.
    - Available equipment operating time: from column 'Available Equipment Operating Time' for each equipment row.
    - Equipment cost at full load: from column 'Equipment Cost at Full Load (yuan)' for each equipment row.
6.  **Formulate Objective:** Maximize total profit, calculated as:
        - Total revenue: sum over all products of (unit price × total quantity produced)
        - Minus total raw material cost: sum over all products of (raw material cost × total quantity produced)
        - Minus total equipment operating cost: sum over all equipment of (equipment cost at full load × (actual equipment usage time / available equipment operating time))
7.  **Formulate Constraints:**
    - Constraint 1 (Procedure Completion): For each product, the total quantity processed in procedure A (across eligible A equipment) must equal the total quantity processed in procedure B (across eligible B equipment), ensuring consistent flow.
    - Constraint 2 (Equipment Assignment Eligibility): Only allow `x[p, e]` > 0 if product `p` is eligible to be processed on equipment `e` for the relevant procedure, as specified in the query.
    - Constraint 3 (Equipment Operating Time Limit): For each equipment, the sum over all assigned products of (processing time per unit × quantity processed) ≤ available equipment operating time.
    - Constraint 4 (Non-negativity): All `x[p, e]` ≥ 0.
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal continuous production plan for three products (I, II, III), each requiring two procedures (A and B), with each procedure performed on specific eligible equipment types. The goal is to maximize profit, considering processing times, raw material costs, selling prices, equipment operating time limits, and equipment costs at full load, as provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) problem with assignment and resource constraints.
3.  **Define Index Sets:** The primary indices are:
    - Products: {I, II, III}
    - Procedures: {A, B}
    - Equipment: {A1, A2, B1, B2, B3} (filtered per product and procedure eligibility)
4.  **Define Decision Variables:**
    - `x[p,e]` = Quantity of product `p` processed on equipment `e` for its respective procedure. Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    - Processing time per unit: from columns 'Product I', 'Product II', 'Product III' for each equipment row.
    - Raw material cost per unit: from 'Raw Material Cost (yuan/unit)' row.
    - Unit selling price: from 'Unit Price (yuan/unit)' row.
    - Available equipment operating time: from 'Available Equipment Operating Time' column.
    - Equipment cost at full load: from 'Equipment Cost at Full Load (yuan)' column.
6.  **Formulate Objective:** Maximize total profit, calculated as:
    - Total revenue: sum over all products of (unit price × total quantity produced of each product)
    - Minus total raw material cost: sum over all products of (raw material cost × total quantity produced)
    - Minus total equipment cost: sum over all equipment of (equipment cost at full load × (actual equipment usage time / available equipment operating time))
7.  **Formulate Constraints:**
    - Constraint 1 (Procedure Completion): For each product and each procedure, the total quantity processed across eligible equipment for that procedure must be equal (i.e., the output of procedure A for a product equals the input to procedure B for that product, and both equal the total production quantity of that product).
    - Constraint 2 (Equipment Eligibility): Only allow `x[p,e]` to be positive if product `p` is eligible to be processed on equipment `e` for the relevant procedure, as specified in the query.
    - Constraint 3 (Equipment Time Limit): For each equipment, the sum over all products of (processing time per unit × quantity processed on that equipment) ≤ available equipment operating time.
    - Constraint 4 (Non-negativity): All `x[p,e]` ≥ 0.
[Abstract Model Plan END]
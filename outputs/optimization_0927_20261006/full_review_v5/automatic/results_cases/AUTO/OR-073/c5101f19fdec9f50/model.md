[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal continuous production plan for three products (I, II, III), each requiring two procedures (A and B), with each procedure performed on specific eligible equipment types. The goal is to maximize profit, considering processing times, raw material costs, selling prices, equipment operating time limits, and equipment costs at full load, as provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) production planning and assignment problem with eligibility constraints.
3.  **Define Index Sets:** The primary indices are:
    - Products: {I, II, III}
    - Procedures: {A, B}
    - Equipment: {A1, A2, B1, B2, B3} (filtered per procedure and product eligibility)
4.  **Define Decision Variables:**
    - `x[p, e]` = Quantity of product `p` processed on equipment `e` for its respective procedure. Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    - Processing time per unit for each (equipment, product) pair: from columns 'Product I', 'Product II', 'Product III' in equipment rows.
    - Raw material cost per unit for each product: from 'Raw Material Cost (yuan/unit)' row.
    - Unit selling price for each product: from 'Unit Price (yuan/unit)' row.
    - Available equipment operating time: from 'Available Equipment Operating Time' column.
    - Equipment cost at full load: from 'Equipment Cost at Full Load (yuan)' column.
6.  **Formulate Objective:** Maximize total profit, calculated as:
        - Total revenue: sum over all products of (unit price × total units produced of each product)
        - Minus total raw material cost: sum over all products of (raw material cost × total units produced)
        - Minus total equipment cost: sum over all equipment of (equipment cost at full load × (actual equipment usage time / available equipment operating time))
7.  **Formulate Constraints:**
    - Constraint 1 (Procedure Completion): For each product, the quantity completed in procedure A (sum over eligible A equipment) must equal the quantity completed in procedure B (sum over eligible B equipment), ensuring consistent flow.
    - Constraint 2 (Equipment Operating Time): For each equipment, the total processing time assigned (sum over eligible products: processing time per unit × quantity assigned) must not exceed its available operating time.
    - Constraint 3 (Eligibility): Only allow assignment of product to equipment if the product is eligible for that equipment and procedure, as specified in the query (e.g., Product III only on A2 for A, and B2 for B).
    - Constraint 4 (Non-negativity): All production quantities `x[p, e]` ≥ 0.
[Abstract Model Plan END]
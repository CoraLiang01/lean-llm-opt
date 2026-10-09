[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal continuous production plan for three products (I, II, III), each requiring two procedures (A and B), with each procedure performed on specific eligible equipment types. The goal is to maximize profit, considering processing times, raw material costs, selling prices, equipment operating time limits, and equipment costs at full load, as provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) production planning and assignment problem with assignment eligibility constraints.
3.  **Define Index Sets:** The primary indices are:
    - Products: {I, II, III}
    - Procedures: {A, B}
    - Equipment: {A1, A2, B1, B2, B3} (filtered per product and procedure eligibility)
4.  **Define Decision Variables:**
    - `x[p,e]` = Quantity of product `p` processed on equipment `e` for its respective procedure. Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    - Processing time per unit for each (product, equipment) pair: from columns 'Product I', 'Product II', 'Product III' in equipment rows.
    - Raw material cost per unit for each product: from 'Raw Material Cost (yuan/unit)' row.
    - Selling price per unit for each product: from 'Unit Price (yuan/unit)' row.
    - Available equipment operating time: from 'Available Equipment Operating Time' column.
    - Equipment cost at full load: from 'Equipment Cost at Full Load (yuan)' column.
6.  **Formulate Objective:** Maximize total profit, calculated as:
    - Total revenue: sum over all products of (selling price per unit × total units produced of that product)
    - Minus total raw material cost: sum over all products of (raw material cost per unit × total units produced)
    - Minus total equipment cost: sum over all equipment of (equipment cost at full load × (total operating time used on equipment / available equipment operating time))
7.  **Formulate Constraints:**
    - **Procedure Completion:** For each product, the quantity produced must be the same across both procedures (i.e., total units processed in procedure A = total units processed in procedure B for each product).
    - **Equipment Assignment Eligibility:** Only allow assignment of product-procedure pairs to eligible equipment as specified:
        - Product I: Procedure A on A1 or A2; Procedure B on B1, B2, or B3.
        - Product II: Procedure A on A1 or A2; Procedure B only on B1.
        - Product III: Procedure A only on A2; Procedure B only on B2.
    - **Equipment Capacity:** For each equipment, the total processing time used (sum over all assigned products: processing time per unit × quantity assigned) ≤ available equipment operating time.
    - **Non-negativity:** All `x[p,e]` ≥ 0.
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal continuous production plan for three products (I, II, III), each requiring two procedures (A and B), with each procedure performed on specific eligible equipment types. The goal is to maximize profit, considering processing times, raw material costs, selling prices, equipment operating time limits, and equipment costs at full load, as detailed in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) problem with assignment constraints (due to product-equipment eligibility).
3.  **Define Index Sets:** The primary indices are:
    - Products: {I, II, III}
    - Procedures: {A, B}
    - Equipment: {A1, A2, B1, B2, B3} (filtered per product and procedure eligibility)
4.  **Define Decision Variables:**
    -   `x[p,e]` = Quantity of product `p` processed on equipment `e` for its respective procedure. Type: GRB.CONTINUOUS.
        - Only define `x[p,e]` for valid (product, equipment) pairs as per eligibility:
            - For Procedure A:
                - Product I: A1, A2
                - Product II: A1, A2
                - Product III: A2 only
            - For Procedure B:
                - Product I: B1, B2, B3
                - Product II: B1 only
                - Product III: B2 only
5.  **Identify Parameters (from Schema):**
    -   Processing time per unit: from columns 'Product I', 'Product II', 'Product III' for each equipment row.
    -   Raw material cost per unit: from 'Raw Material Cost (yuan/unit)' row.
    -   Unit selling price: from 'Unit Price (yuan/unit)' row.
    -   Available equipment operating time: from 'Available Equipment Operating Time' column.
    -   Equipment cost at full load: from 'Equipment Cost at Full Load (yuan)' column.
6.  **Formulate Objective:** Maximize total profit, calculated as:
    - Total revenue: sum over all products of (unit price × total units produced of each product)
    - Minus total raw material cost: sum over all products of (raw material cost × total units produced)
    - Minus total equipment operating cost: sum over all equipment of (equipment cost at full load × (actual operating time used / available operating time))
    - Where actual operating time used per equipment is the sum over all assigned products of (processing time per unit × quantity produced on that equipment)
7.  **Formulate Constraints:**
    -   **Procedure Completion:** For each product, the quantity processed in procedure A must equal the quantity processed in procedure B (i.e., production is synchronized across procedures).
        - For each product p: sum over eligible A equipment of x[p,e] (A) = sum over eligible B equipment of x[p,e] (B)
    -   **Equipment Operating Time Limits:** For each equipment, the total processing time assigned cannot exceed its available operating time.
        - For each equipment e: sum over eligible products p of (processing time per unit × x[p,e]) ≤ available operating time for e
    -   **Eligibility Constraints:** Only allow variables x[p,e] for valid (product, equipment) pairs as per the eligibility rules.
    -   **Non-negativity:** All x[p,e] ≥ 0 (continuous).
[Abstract Model Plan END]
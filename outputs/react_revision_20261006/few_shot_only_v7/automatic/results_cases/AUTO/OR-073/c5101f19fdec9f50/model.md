[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal continuous production plan for three products (I, II, III), each requiring two procedures (A and B), with each procedure performed on specific eligible equipment types. The goal is to maximize profit, considering processing times, raw material costs, selling prices, equipment operating time limits, and equipment costs at full load, as specified in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) production planning and assignment problem with assignment restrictions.
3.  **Define Index Sets:** The primary indices are:
    - Products: {I, II, III}
    - Procedures: {A, B}
    - Equipment: {A1, A2, B1, B2, B3}
      (Note: Only equipment explicitly mentioned in the query for each product/procedure are included.)
4.  **Define Decision Variables:**
    -   `x[p,e]` = Quantity of product `p` processed on equipment `e` for its respective procedure (continuous, GRB.CONTINUOUS). Only define variables for allowed (product, equipment) pairs as per the query.
5.  **Identify Parameters (from Schema):**
    -   Processing time per unit: from the intersection of [Equipment / Cost] row and [Product] column (e.g., schema['A1']['Product I']).
    -   Raw material cost per unit: from row 'Raw Material Cost (yuan/unit)' and product columns.
    -   Selling price per unit: from row 'Unit Price (yuan/unit)' and product columns.
    -   Equipment available operating time: from 'Available Equipment Operating Time' column for each equipment.
    -   Equipment cost at full load: from 'Equipment Cost at Full Load (yuan)' for each equipment.
6.  **Formulate Objective:** Maximize total profit, calculated as:
    - Total revenue: sum over all products of (selling price per unit × total units produced of each product)
    - Minus total raw material cost: sum over all products of (raw material cost per unit × total units produced)
    - Minus total equipment cost: sum over all equipment of (equipment cost at full load × (actual operating time used / available operating time))
    - Where actual operating time used per equipment is the sum over all assigned products of (processing time per unit × quantity produced on that equipment).
7.  **Formulate Constraints:**
    -   **Procedure Assignment Constraints:** For each product and procedure, production must be assigned only to eligible equipment as specified:
        - Product I: Procedure A on A1 or A2; Procedure B on B1, B2, or B3.
        - Product II: Procedure A on A1 or A2; Procedure B only on B1.
        - Product III: Procedure A only on A2; Procedure B only on B2.
    -   **Flow Conservation Constraints:** For each product, the quantity completed in procedure A (sum over eligible A equipment) must equal the quantity completed in procedure B (sum over eligible B equipment), ensuring consistent production flow.
    -   **Equipment Operating Time Constraints:** For each equipment, the total processing time used (sum over all assigned products: processing time per unit × quantity produced) must not exceed the available equipment operating time.
    -   **Non-negativity:** All production quantities `x[p,e]` ≥ 0.
[Abstract Model Plan END]
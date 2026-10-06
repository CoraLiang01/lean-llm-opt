[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal continuous production quantities of three products (I, II, III) in a factory, considering that each product must be processed through two procedures (A and B), with each procedure having specific eligible equipment types per product. The goal is to maximize profit, accounting for raw material costs, equipment operating costs, processing times, equipment capacities, and selling prices, as detailed in the provided CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) production planning and assignment problem with assignment restrictions.
3.  **Define Index Sets:** The primary indices are:
    - Products: {I, II, III}
    - Equipment: {A1, A2, B1, B2, B3} (filtered per procedure and product eligibility)
    - Procedures: {A, B}
4.  **Define Decision Variables:**
    - `q[p]` = Quantity of product p produced (for p in {I, II, III}). Type: GRB.CONTINUOUS.
    - `x[e,p]` = Amount of product p processed on equipment e (for eligible (e,p) pairs). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    - Processing time per unit: from columns 'Product I', 'Product II', 'Product III' for each equipment row.
    - Raw material cost per unit: from row 'Raw Material Cost (yuan/unit)'.
    - Unit selling price: from row 'Unit Price (yuan/unit)'.
    - Available equipment operating time: from column 'Available Equipment Operating Time'.
    - Equipment cost at full load: from column 'Equipment Cost at Full Load (yuan)' (to be allocated proportionally to usage).
6.  **Formulate Objective:** Maximize total profit, defined as:
    - Total revenue: sum over products of (unit price * quantity produced)
    - Minus total raw material cost: sum over products of (raw material cost * quantity produced)
    - Minus total equipment operating cost: sum over all equipment of (equipment cost at full load * (actual operating time used / available operating time))
7.  **Formulate Constraints:**
    - **Procedure Assignment Constraints:** For each product and procedure, ensure that the total quantity produced is fully assigned to eligible equipment for that procedure (e.g., for procedure A of product I, sum of x[e,I] over eligible A equipment = q[I]).
    - **Equipment Capacity Constraints:** For each equipment, the total processing time used (sum over all assigned products: processing time per unit * x[e,p]) must not exceed the available equipment operating time.
    - **Eligibility Constraints:** Only allow x[e,p] > 0 if equipment e is eligible to process product p for the relevant procedure (as specified in the query).
    - **Flow Consistency Constraints:** For each product, the quantity processed in procedure A must equal the quantity processed in procedure B (i.e., q[p] is consistent across both procedures).
    - **Non-negativity:** All decision variables must be non-negative.
[Abstract Model Plan END]
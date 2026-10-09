[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal continuous production quantities of three products (I, II, III) in a factory, considering that each product must be processed through two procedures (A and B), with each procedure having specific eligible equipment types. The goal is to maximize profit, accounting for raw material costs, equipment operating costs, processing times, equipment capacities, and product selling prices, as provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) production planning and assignment problem with assignment restrictions.
3.  **Define Index Sets:** The primary indices are:
    - Products: {I, II, III}
    - Equipment: {A1, A2} for Procedure A; {B1, B2, B3} for Procedure B (as per query, only these are relevant)
    - Procedures: {A, B}
4.  **Define Decision Variables:**
    - `x[p]` = Quantity of product p produced (for p in {I, II, III}). Type: GRB.CONTINUOUS.
    - `y[e,p]` = Amount of product p processed on equipment e (for eligible (e,p) pairs, as per assignment rules). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    - Processing time per unit: from columns 'Product I', 'Product II', 'Product III' for each equipment row.
    - Raw material cost per unit: from row 'Raw Material Cost (yuan/unit)'.
    - Unit selling price: from row 'Unit Price (yuan/unit)'.
    - Available equipment operating time: from column 'Available Equipment Operating Time'.
    - Equipment cost at full load: from column 'Equipment Cost at Full Load (yuan)' (to be allocated proportionally to usage).
6.  **Formulate Objective:** Maximize total profit, defined as:
    - Total revenue: sum over products of (unit price * quantity produced)
    - Minus total raw material cost: sum over products of (raw material cost * quantity produced)
    - Minus total equipment operating cost: sum over all equipment of (equipment cost at full load * (total time used on equipment / available equipment operating time))
    - Objective: Maximize [sum_p (unit price_p - raw material cost_p) * x[p]] - sum_e (equipment cost at full load_e * (total time used_e / available time_e))
7.  **Formulate Constraints:**
    - **Procedure Assignment Constraints:** For each product and procedure, ensure that the product is only processed on eligible equipment:
        - Product I: Procedure A on A1 or A2; Procedure B on B1, B2, or B3.
        - Product II: Procedure A on A1 or A2; Procedure B only on B1.
        - Product III: Procedure A only on A2; Procedure B only on B2.
    - **Production-Processing Consistency:** For each product and procedure, the total amount processed across eligible equipment must equal the production quantity:
        - For each product p and procedure s: sum over eligible equipment e of y[e,p] = x[p]
    - **Equipment Capacity Constraints:** For each equipment, the total processing time used cannot exceed its available operating time:
        - For each equipment e: sum over products p of (processing time per unit for (e,p) * y[e,p]) ≤ available equipment operating time for e
    - **Non-negativity:** All decision variables must be ≥ 0.
[Abstract Model Plan END]
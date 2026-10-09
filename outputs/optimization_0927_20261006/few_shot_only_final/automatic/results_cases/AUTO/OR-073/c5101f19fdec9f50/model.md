[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal continuous production plan for three products (I, II, III), each requiring two procedures (A and B), with each procedure performed on specific eligible equipment types. The goal is to maximize profit, considering processing times, raw material costs, selling prices, equipment operating time limits, and equipment costs at full load, as provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) production planning and assignment problem with assignment restrictions.
3.  **Define Index Sets:** The primary indices are:
    - Products: {I, II, III}
    - Procedures: {A, B}
    - Equipment: {A1, A2, B1, B2, B3}
4.  **Define Decision Variables:**
    - `x[p,e]` = Quantity of product `p` processed on equipment `e` for its respective procedure. Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - Selling price per unit: from row with "Unit Price (yuan/unit)" for each product.
        - Raw material cost per unit: from row with "Raw Material Cost (yuan/unit)" for each product.
        - Equipment cost at full load: from "Equipment Cost at Full Load (yuan)" for each equipment.
    - Constraint coefficients:
        - Processing time per unit: from each equipment row, columns "Product I", "Product II", "Product III".
        - Available equipment operating time: from "Available Equipment Operating Time" for each equipment.
    - Constraint RHS:
        - Equipment operating time limits: from "Available Equipment Operating Time".
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products and equipment of (selling price per unit - raw material cost per unit) times the quantity produced, minus the sum over all equipment of the proportional equipment cost incurred (equipment cost at full load times the fraction of equipment time used).
7.  **Formulate Constraints:**
    - Constraint 1 (Procedure Assignment): For each product and procedure, production must be assigned only to eligible equipment as specified:
        - Product I: Procedure A on A1 or A2; Procedure B on B1, B2, or B3.
        - Product II: Procedure A on A1 or A2; Procedure B only on B1.
        - Product III: Procedure A only on A2; Procedure B only on B2.
    - Constraint 2 (Procedure Flow): For each product, the total quantity processed in procedure A must equal the total quantity processed in procedure B (i.e., production continuity across procedures).
    - Constraint 3 (Equipment Capacity): For each equipment, the total processing time used (sum over all assigned products: processing time per unit × quantity) must not exceed the available equipment operating time.
    - Constraint 4 (Non-negativity): All production quantities `x[p,e]` ≥ 0.
[Abstract Model Plan END]
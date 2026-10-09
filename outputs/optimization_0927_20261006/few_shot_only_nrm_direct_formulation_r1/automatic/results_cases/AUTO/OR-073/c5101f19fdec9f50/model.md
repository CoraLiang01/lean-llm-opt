[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal continuous production quantities of three products (I, II, III) across two procedures (A and B), each with specific eligible equipment, to maximize total profit. The model must account for processing times, raw material costs, selling prices, equipment operating time limits, and equipment costs at full load, as provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) production planning and assignment problem with eligibility constraints.
3.  **Define Index Sets:** The primary indices are:
    - Products: {I, II, III}
    - Procedures: {A, B}
    - Equipment: {A1, A2, B1, B2, B3} (restricted per product and procedure as described)
4.  **Define Decision Variables:**
    - `x[p,e]` = Quantity of product `p` processed on equipment `e` (continuous, GRB.CONTINUOUS), for all eligible (product, equipment) pairs as per procedure and eligibility.
5.  **Identify Parameters (from Schema):**
    - Processing time per unit: from columns "Product I", "Product II", "Product III" for each equipment row.
    - Raw material cost per unit: from row "Raw Material Cost (yuan/unit)" for each product.
    - Selling price per unit: from row "Unit Price (yuan/unit)" for each product.
    - Available equipment operating time: from column "Available Equipment Operating Time" for each equipment.
    - Equipment cost at full load: from column "Equipment Cost at Full Load (yuan)" for each equipment.
6.  **Formulate Objective:** Maximize total profit, defined as:
        - Total revenue: sum over all products and equipment of (selling price per unit * total units produced of each product)
        - Minus total raw material cost: sum over all products and equipment of (raw material cost per unit * total units produced)
        - Minus total equipment cost: sum over all equipment of (equipment cost at full load * (total time used on equipment / available equipment operating time))
7.  **Formulate Constraints:**
    - Constraint 1 (Equipment Operating Time): For each equipment, the sum over all products of (processing time per unit * quantity assigned to that equipment) ≤ available equipment operating time.
    - Constraint 2 (Procedure Assignment Eligibility): Only allow assignment of product to equipment if eligible (e.g., Product III only on A2 for procedure A, and only on B2 for procedure B; Product II only on B1 for procedure B, etc.).
    - Constraint 3 (Procedure Completion): For each product, the quantity produced must be the same across both procedures (i.e., total units assigned to eligible A equipment = total units assigned to eligible B equipment for each product).
    - Constraint 4 (Non-negativity): All `x[p,e]` ≥ 0.
[Abstract Model Plan END]
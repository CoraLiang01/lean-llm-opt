[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal continuous production quantities of three products (I, II, III) in a factory, where each product must be processed through two procedures (A and B) using specific types of equipment. The goal is to maximize profit, considering raw material costs, selling prices, processing times, available equipment operating times, and equipment operating costs. Each product can only be processed on certain equipment types for each procedure, as specified in the query.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) production planning problem with assignment constraints.
3.  **Define Index Sets:** The primary indices are:
    - Products: {I, II, III}
    - Equipment: {A1, A2, B1, B2, B3} (filtered per product and procedure as per query)
    - Procedures: {A, B}
4.  **Define Decision Variables:**
    -   `q[p]` = Quantity of product p produced (for p in {I, II, III}). Type: GRB.CONTINUOUS.
    -   `proc_time[p,e]` = Processing time allocated to product p on equipment e (for feasible (p,e) pairs). Type: GRB.CONTINUOUS.
        - (Alternatively, since all production is continuous and assignment is fixed by feasibility, the model may directly allocate production to equipment where allowed.)
5.  **Identify Parameters (from Schema):**
    -   Processing time per unit for each (product, equipment) pair: from columns 'Product I', 'Product II', 'Product III' in equipment rows.
    -   Raw material cost per unit: from 'Raw Material Cost (yuan/unit)' row.
    -   Unit selling price: from 'Unit Price (yuan/unit)' row.
    -   Available equipment operating time: from 'Available Equipment Operating Time' column.
    -   Equipment cost at full load: from 'Equipment Cost at Full Load (yuan)' column.
6.  **Formulate Objective:** Maximize total profit, defined as:
    -   Total revenue: sum over products of (unit price * quantity produced)
    -   Minus total raw material cost: sum over products of (raw material cost * quantity produced)
    -   Minus total equipment operating cost: sum over equipment of (equipment cost at full load * (actual equipment usage / full load time))
    -   Objective: Maximize [sum_p (unit price_p - raw material cost_p) * q[p]] - sum_e (equipment cost at full load_e * (total time used on e / available time_e))
7.  **Formulate Constraints:**
    -   Constraint 1 (Equipment Time Limits): For each equipment e, the total processing time assigned to all products on e cannot exceed its available operating time.
        - sum_p (processing time per unit for (p,e) * q[p]) ≤ available equipment operating time for e
        - Only include (p,e) pairs where the product is allowed on the equipment for the relevant procedure.
    -   Constraint 2 (Procedure Assignment Feasibility): Only allow products to be processed on equipment types as specified:
        - Product I: Procedure A on A1 or A2; Procedure B on B1, B2, or B3
        - Product II: Procedure A on A1 or A2; Procedure B only on B1
        - Product III: Procedure A only on A2; Procedure B only on B2
    -   Constraint 3 (Non-negativity): All production quantities and equipment usage variables must be ≥ 0.
[Abstract Model Plan END]
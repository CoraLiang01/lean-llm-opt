[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal continuous production quantities of three products (I, II, III) on eligible equipment for two procedures (A and B), maximizing total profit. Each product has specific eligibility for equipment types in each procedure. The model must account for processing times, raw material costs, selling prices, available equipment operating times, and equipment costs at full load, as provided in the CSV.
2.  **Identify Model Type:** Based on the query, this is a Linear Programming (LP) production planning problem with assignment constraints.
3.  **Define Index Sets:** The primary indices are:
    - Products: {I, II, III}
    - Procedures: {A, B}
    - Equipment: {A1, A2, B1, B2, B3} (restricted to those eligible for each product/procedure as per query)
4.  **Define Decision Variables:**
    -   `x[p,e]` = Quantity of product `p` processed on equipment `e` (continuous, GRB.CONTINUOUS), for all eligible (product, equipment) pairs as per procedure.
5.  **Identify Parameters (from Schema):**
    -   Processing time per unit: from columns under each product for each equipment row.
    -   Raw material cost per unit: from "Raw Material Cost (yuan/unit)" row for each product.
    -   Unit selling price: from "Unit Price (yuan/unit)" row for each product.
    -   Available equipment operating time: from "Available Equipment Operating Time" column for each equipment.
    -   Equipment cost at full load: from "Equipment Cost at Full Load (yuan)" column for each equipment.
6.  **Formulate Objective:** Maximize total profit, calculated as:
        - Total revenue: sum over all products and equipment of (unit price × total quantity produced of each product)
        - Minus total raw material cost: sum over all products and equipment of (raw material cost × total quantity produced)
        - Minus total equipment cost: sum over all equipment of (equipment cost at full load × (total equipment time used / available equipment operating time))
7.  **Formulate Constraints:**
    -   **Equipment Time Limits:** For each equipment, the sum over all eligible products of (processing time per unit × quantity produced) ≤ available equipment operating time.
    -   **Procedure Completion:** For each product, the total quantity produced in procedure A (sum over eligible A equipment) must equal the total quantity produced in procedure B (sum over eligible B equipment), ensuring each unit passes through both procedures.
    -   **Eligibility Constraints:** Only allow variables `x[p,e]` for (product, equipment) pairs permitted by the query (e.g., Product III only on A2 for procedure A and B2 for procedure B; Product II only on B1 for procedure B, etc.).
    -   **Non-negativity:** All production quantities `x[p,e]` ≥ 0.
[Abstract Model Plan END]
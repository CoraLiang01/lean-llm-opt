[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select integer quantities of authorized vehicle configurations for RIVERSIDE_AUTO’s next delivery, maximizing total net benefit (unit benefit minus fixed preparation charges), while respecting resource limits, category quantity bounds, item and category activation fees, incompatibilities, requires dependencies, and bundle bonuses. All data is to be taken directly from the supplied tables.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, logical (binary) constraints, and combinatorial bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items (vehicle configurations) from the 'item' table (export_06.csv).
    - Categories from the 'category' table (export_03.csv).
    - Resources from the 'capacity_ledger' and 'usage' tables (export_02.csv, export_09.csv).
    - Bundles (pairs of items) from the 'bundle' table (export_01.csv).
    - Incompatible pairs from the 'incompatible' table (export_05.csv).
    - Requires dependencies from the 'requires' table (export_08.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity ordered of item i (vehicle configuration). Type: GRB.INTEGER. Must be 0 if unauthorized; otherwise, between minimum_lot and maximum_order.
    -   `y[i]` = Binary variable: 1 if item i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = Binary variable: 1 if any item in category c is selected (i.e., sum over i in c of y[i] ≥ 1), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = Binary variable: 1 if both items in bundle are selected (i.e., both y[i_a] and y[i_b] are 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   'unit_benefit_cents' (per unit, from 'item' table/export_06.csv).
        -   'item_fee_cents' (fixed per item if used, from 'item' table/export_06.csv).
        -   'activation_fee_cents' (fixed per category if used, from 'category' table/export_03.csv).
        -   'bonus_cents' (per bundle, from 'bundle' table/export_01.csv).
    -   Constraint coefficients:
        -   'usage' per item per resource (from 'usage' table/export_09.csv).
        -   Resource limits (sum of 'opening' and 'reservation' per resource from 'capacity_ledger' table/export_02.csv).
        -   Category minimum/maximum quantities (from 'category' table/export_03.csv).
        -   Item minimum_lot and maximum_order (from 'item' table/export_06.csv).
        -   Authorization status (from 'item' table/export_06.csv).
        -   Incompatible pairs (from 'incompatible' table/export_05.csv).
        -   Requires dependencies (from 'requires' table/export_08.csv).
    -   Index mappings:
        -   Item-to-category mapping (from 'item' table/export_06.csv).
        -   Bundle item pairs (from 'bundle' table/export_01.csv).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over all items: (item_fee_cents[i] * y[i]) [charged once per item if any units ordered]
    -   Minus sum over all categories: (activation_fee_cents[c] * z[c]) [charged once per category if any item in c is ordered]
    -   Plus sum over all bundles: (bonus_cents[bundle] * b[bundle]) [awarded once per bundle if both items are ordered]
7.  **Formulate Constraints:**
    -   **Item Authorization and Lot Size:**
        -   For each item i: If authorized[i] == 0, x[i] = 0.
        -   For each item i: If authorized[i] == 1, x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i].
    -   **Item Activation Linking:**
        -   For each item i: y[i] = 1 if x[i] ≥ 1, else y[i] = 0. (Enforced via: x[i] ≤ maximum_order[i] * y[i], x[i] ≥ minimum_lot[i] * y[i] or x[i] = 0)
    -   **Category Activation Linking:**
        -   For each category c: z[c] = 1 if any y[i] for i in c is 1, else 0. (Enforced via: For all i in c, y[i] ≤ z[c]; and sum over i in c of y[i] ≥ z[c])
    -   **Category Quantity Bounds:**
        -   For each category c: minimum_quantity[c] ≤ sum over i in c of x[i] ≤ maximum_quantity[c]
    -   **Resource Capacity Constraints:**
        -   For each resource r: sum over all items i of (usage[i, r] * x[i]) ≤ total_capacity[r], where total_capacity[r] = sum of 'opening' and 'reservation' for r from 'capacity_ledger'
    -   **Incompatibility Constraints:**
        -   For each incompatible pair (i, j): y[i] + y[j] ≤ 1 (cannot select both)
    -   **Requires Dependencies:**
        -   For each requires pair (i, prereq): y[i] ≤ y[prereq] (can only select i if prereq is also selected)
        -   Alternatively, x[i] ≤ maximum_order[i] * y[prereq]
    -   **Bundle Bonuses:**
        -   For each bundle (i_a, i_b): b[bundle] ≤ y[i_a], b[bundle] ≤ y[i_b], b[bundle] ≥ y[i_a] + y[i_b] - 1 (b[bundle] = 1 iff both items are selected)
        -   If either item in a bundle is unauthorized, b[bundle] = 0.
    -   **Variable Domains:**
        -   x[i]: integer, ≥ 0
        -   y[i], z[c], b[bundle]: binary (0 or 1)
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select integer quantities of authorized vehicle configurations for RIVERSIDE_AUTO’s next delivery, maximizing total net benefit (unit benefit minus fixed preparation/item fees and category activation fees, plus bundle bonuses), subject to resource, category, compatibility, and dependency constraints. Only authorized options may be ordered, and each has minimum/maximum lot sizes. Resource and category limits, incompatibilities, and requires dependencies must be respected. Bundle bonuses are awarded once per eligible pair.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) and logical (compatibility/dependency) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (vehicle configurations) — from all rows in the 'item' table (export_06.csv).
    - Categories — from all rows in the 'category' table (export_03.csv).
    - Resources — from all rows in the 'capacity_ledger' and 'usage' tables (export_02.csv, export_09.csv).
    - Bundles — from all rows in the 'bundle' table (export_01.csv).
    - Incompatible pairs — from all rows in the 'incompatible' table (export_05.csv).
    - Requires pairs — from all rows in the 'requires' table (export_08.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item (configuration) `i` to order. Type: GRB.INTEGER. Must be zero if not authorized.
    -   `z[i]` = Binary variable: 1 if item `i` is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[c]` = Binary variable: 1 if any item in category `c` is selected (i.e., sum over i in c of z[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = Binary variable: 1 if both items in bundle are selected (i.e., both z[i_a] and z[i_b] are 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents` (per unit, from 'item' table/export_06.csv).
        -   `item_fee_cents` (fixed per item if any ordered, from 'item' table/export_06.csv).
        -   `activation_fee_cents` (fixed per category if any item in category is ordered, from 'category' table/export_03.csv).
        -   `bonus_cents` (per bundle, from 'bundle' table/export_01.csv).
    -   Constraint coefficients:
        -   `usage` per item per resource (from 'usage' table/export_09.csv).
        -   Resource limits (sum of 'opening' and 'reservation' for each resource from 'capacity_ledger' table/export_02.csv).
        -   Category min/max quantities (from 'category' table/export_03.csv).
        -   Item min/max lot sizes (from 'item' table/export_06.csv).
        -   Authorization status (from 'item' table/export_06.csv).
        -   Incompatible pairs (from 'incompatible' table/export_05.csv).
        -   Requires pairs (from 'requires' table/export_08.csv).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over all items: (item_fee_cents[i] * z[i]) [fixed fee if any of item i is ordered]
    -   Minus sum over all categories: (activation_fee_cents[c] * w[c]) [fixed fee if any item in category c is ordered]
    -   Plus sum over all bundles: (bonus_cents[bundle] * b[bundle]) [bonus if both items in bundle are selected]
7.  **Formulate Constraints:**
    -   **Authorization:** For each item i, if authorized[i] == 0, then x[i] = 0 and z[i] = 0.
    -   **Lot Size:** For each item i, if authorized[i] == 1, then either x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i].
    -   **Linking x and z:** For each item i, x[i] ≥ minimum_lot[i] * z[i] and x[i] ≤ maximum_order[i] * z[i].
    -   **Resource Limits:** For each resource r, sum over items i of (usage[i, r] * x[i]) ≤ total available for r (sum of 'opening' and 'reservation' for r).
    -   **Category Quantity Limits:** For each category c, sum over items i in c of x[i] ≥ minimum_quantity[c] * w[c] and ≤ maximum_quantity[c] * w[c]; also, w[c] = 1 if any x[i] > 0 for i in c, else 0.
    -   **Incompatibility:** For each incompatible pair (i, j), z[i] + z[j] ≤ 1 (cannot select both).
    -   **Requires Dependencies:** For each requires pair (i, prereq), z[i] ≤ z[prereq] (cannot select i unless prereq is also selected).
    -   **Bundle Bonuses:** For each bundle (i_a, i_b), b[bundle] ≤ z[i_a], b[bundle] ≤ z[i_b], b[bundle] ≥ z[i_a] + z[i_b] - 1 (b[bundle] = 1 iff both items are selected).
    -   **Variable Domains:** x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer), z[i], w[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]
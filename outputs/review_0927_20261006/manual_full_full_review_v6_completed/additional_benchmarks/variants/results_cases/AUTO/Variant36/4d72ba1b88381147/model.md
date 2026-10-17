[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer quantities of authorized air-conditioner models (item_ref) to assign to storage areas (location_id) at FC_EAST_HVAC, maximizing net value (total benefit plus bonuses minus all fixed and variable fees), subject to per-area volume limits, per-category aggregate quantity bounds, item compatibility and dependency rules, and bundle bonuses. Each area’s capacity is independent, and only authorized options may be selected.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, group activation, and logical (compatibility, dependency, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items: All item_ref rows from the union of 'item' tables (from batch_06/export_06.csv and batch_01/export_07.csv) where authorized = 1.
    - Categories: All category rows from the 'category' table (batch_03/export_03.csv).
    - Resources: All resource rows from the 'capacity_ledger' and 'usage' tables (batch_02/export_02.csv, batch_04/export_10.csv, batch_05/export_11.csv).
    - Bundles: All bundle rows from the 'bundle' table (batch_01/export_01.csv).
    - Incompatibilities: All pairs from the 'incompatible' table (batch_05/export_05.csv).
    - Requirements: All pairs from the 'requires' table (batch_03/export_09.csv).
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity of item i selected (0 or in [minimum_lot, maximum_order] for authorized items; 0 for unauthorized). Type: GRB.INTEGER.
    -   `z[i]` = 1 if item i is selected (q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `g[c]` = 1 if any item in category c is selected (i.e., sum_{i in c} z[i] ≥ 1), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected (z[item_a] = z[item_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   unit_benefit_cents (from 'item' tables) for each item_ref.
        -   item_fee_cents (from 'item' tables) for each item_ref (fixed fee if selected).
        -   activation_fee_cents (from 'category' table) for each category (fixed fee if any item in category is selected).
        -   bonus_cents (from 'bundle' table) for each bundle (applied if both items are selected).
    -   Constraint coefficients:
        -   usage amount per item/resource (from 'usage' tables, batch_04/export_10.csv and batch_05/export_11.csv).
        -   capacity_ledger amounts per resource (sum of 'opening' and 'reservation' entries from batch_02/export_02.csv).
        -   minimum_lot, maximum_order per item (from 'item' tables).
        -   minimum_quantity, maximum_quantity per category (from 'category' table).
        -   authorized flag per item (from 'item' tables).
        -   incompatible pairs (from 'incompatible' table).
        -   requires pairs (from 'requires' table).
6.  **Formulate Objective:** Maximize total net benefit in cents:
        -   Sum over all items: (unit_benefit_cents[i] * q[i]) 
        -   Minus sum over all items: (item_fee_cents[i] * z[i]) 
        -   Minus sum over all categories: (activation_fee_cents[c] * g[c])
        -   Plus sum over all bundles: (bonus_cents[bundle] * b[bundle])
7.  **Formulate Constraints:**
    -   **Item selection and bounds:** For each item i:
        -   If authorized[i] = 1: q[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]]; enforce q[i] ≥ minimum_lot[i] * z[i], q[i] ≤ maximum_order[i] * z[i], and q[i] = 0 if z[i] = 0.
        -   If authorized[i] = 0: q[i] = 0, z[i] = 0.
    -   **Resource (area) capacity:** For each resource r (e.g., AREA_A, AREA_B, AREA_C):
        -   Sum over all items assigned to r: (usage amount of i in r) * q[i] ≤ total available capacity for r (sum of 'opening' and 'reservation' in capacity_ledger for r).
    -   **Category aggregate bounds:** For each category c:
        -   Sum over all items in c: q[i] ≥ minimum_quantity[c]
        -   Sum over all items in c: q[i] ≤ maximum_quantity[c]
        -   g[c] = 1 if any z[i] = 1 for i in c; g[c] = 0 otherwise.
    -   **Incompatibility:** For each incompatible pair (i, j):
        -   z[i] + z[j] ≤ 1 (cannot select both).
    -   **Requires:** For each requires pair (i, prereq):
        -   z[i] ≤ z[prereq] (if i is selected, prereq must also be selected).
    -   **Bundle bonuses:** For each bundle (item_a, item_b):
        -   b[bundle] ≤ z[item_a], b[bundle] ≤ z[item_b], b[bundle] ≥ z[item_a] + z[item_b] - 1 (b[bundle] = 1 iff both items are selected).
    -   **Variable domains:** All q[i] are integer, all z[i], g[c], b[bundle] are binary.
[Abstract Model Plan END]
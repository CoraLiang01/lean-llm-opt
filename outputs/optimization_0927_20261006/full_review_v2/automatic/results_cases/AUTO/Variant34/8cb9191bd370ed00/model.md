[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of packs to display for each authorized product-and-section option (item_ref), maximizing total net merchandising benefit (in USD cents), subject to section capacity, per-option and per-category limits, incompatibility and prerequisite rules, and bundle bonuses. The plan must deduct fixed item and category activation fees as specified, and only allow selection of authorized options within their minimum and maximum order bounds.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, requires, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Options: All item_ref values from the union of both item tables (export_07.csv and export_08.csv), each representing a product-section configuration.
    - Sections: Unique location_id values from the item tables.
    - Categories: Unique category values from the item tables and category table.
    - Resources: Unique resource values from the usage and capacity_ledger tables.
    - Bundles: Pairs (item_a, item_b) from the bundle table.
    - Incompatibilities: Pairs (item_a, item_b) from the incompatible table.
    - Requires: Pairs (item_ref, prerequisite_ref) from the requires table.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of packs of option i (item_ref) to display. Type: GRB.INTEGER, with bounds [0, maximum_order[i]], and x[i] = 0 if not authorized.
    -   `z[i]` = 1 if option i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[c]` = 1 if any option in category c is selected (sum over i in c of z[i] ≥ 1), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both options in bundle are selected (z[item_a] = z[item_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each option: sum of amount_cents for each item_ref in the benefit table (export_01.csv).
    -   Item activation fee: activation_fee_cents from item_fee table (export_09.csv), applied once per option if selected.
    -   Category activation fee: activation_fee_cents from category table (export_04.csv), applied once per category if any option in that category is selected.
    -   Section capacity: sum of amount (converted to ml if needed) for each resource in capacity_ledger table (export_03.csv).
    -   Per-pack resource usage: amount (converted to ml if needed) for each (item_ref, resource) in usage tables (export_12.csv and export_13.csv).
    -   Option bounds: minimum_lot and maximum_order from item tables (export_07.csv and export_08.csv), and authorized flag.
    -   Category quantity limits: minimum_quantity and maximum_quantity from category table (export_04.csv).
    -   Bundle bonuses: bonus_cents from bundle table (export_02.csv).
    -   Incompatibility and requires relationships: from incompatible (export_06.csv) and requires (export_11.csv) tables.
6.  **Formulate Objective:** Maximize total net merchandising benefit, defined as:
        - Sum over all options of (per-unit benefit * x[i])
        - Minus sum over all selected options of item activation_fee_cents * z[i]
        - Minus sum over all used categories of category activation_fee_cents * w[c]
        - Plus sum over all bundles of bonus_cents * b[bundle]
7.  **Formulate Constraints:**
    -   Option selection and bounds: For each option i, x[i] = 0 if not authorized; otherwise, minimum_lot[i] ≤ x[i] ≤ maximum_order[i] or x[i] = 0. z[i] = 1 if x[i] ≥ 1, 0 otherwise.
    -   Section (resource) capacity: For each resource r (section), sum over all options using r of (per-pack usage[i,r] * x[i]) ≤ total available capacity[r] (sum of capacity_ledger entries for r, with all units in ml).
    -   Category quantity limits: For each category c, sum over all options in c of x[i] ≥ minimum_quantity[c] and ≤ maximum_quantity[c].
    -   Category activation: w[c] = 1 if any x[i] > 0 for i in c, 0 otherwise.
    -   Incompatibility: For each incompatible pair (i, j), z[i] + z[j] ≤ 1.
    -   Requires: For each (i, prereq), z[i] ≤ z[prereq].
    -   Bundle bonuses: For each bundle (item_a, item_b), b[bundle] = 1 if z[item_a] = z[item_b] = 1, 0 otherwise.
    -   Integrality: All x[i] are integer, all z[i], w[c], b[bundle] are binary.
[Abstract Model Plan END]
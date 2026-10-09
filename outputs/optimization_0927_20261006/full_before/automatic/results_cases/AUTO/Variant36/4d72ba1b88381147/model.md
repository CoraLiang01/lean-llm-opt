[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer quantities of authorized air-conditioner models (item_ref) to assign to storage areas (location_id) at FC_EAST_HVAC, maximizing net value (total benefit minus all fixed and variable fees), while respecting per-area volume limits, per-category quantity bounds, item compatibility, prerequisite requirements, and bundle bonuses. Each area’s capacity is independent, and only authorized options may be chosen.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, assignment, and logical (compatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items: All item_ref rows from the supplied item tables (export_06.csv and export_07.csv).
    - Categories: All category rows from export_03.csv.
    - Resources/Areas: All resource/location_id values from capacity_ledger and usage tables (AREA_A, AREA_B, AREA_C).
    - Bundles: All bundle rows from export_01.csv.
    - Incompatibilities: All incompatible pairs from export_05.csv.
    - Prerequisites: All requires pairs from export_09.csv.
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity of item_ref i to assign (0 if not chosen). Type: GRB.INTEGER. Domain: 0 or any integer between minimum_lot[i] and maximum_order[i] (if authorized), else 0.
    -   `z[i]` = 1 if item_ref i is selected (i.e., q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[c]` = 1 if any item in category c is selected (i.e., sum of q[i] for items in c > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected (i.e., both z[item_a] and z[item_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        - unit_benefit_cents (from item tables) for each item_ref.
        - item_fee_cents (from item tables) for each item_ref (incurred once if any quantity is chosen).
        - activation_fee_cents (from category table) for each category (incurred once if any item in the category is chosen).
        - bonus_cents (from bundle table) for each bundle (added if both items in the bundle are selected).
    -   Constraint coefficients:
        - usage amount per item_ref and resource (from usage tables).
        - capacity_ledger amounts per resource (sum opening and reservation for each area).
        - minimum_lot and maximum_order per item_ref (from item tables).
        - minimum_quantity and maximum_quantity per category (from category table).
        - authorized flag per item_ref (from item tables).
        - incompatible pairs (from incompatible table).
        - prerequisite pairs (from requires table).
    -   Constraint RHS:
        - Per-area (resource) capacity: sum of capacity_ledger entries for each resource.
        - Per-category quantity bounds: minimum_quantity and maximum_quantity.
6.  **Formulate Objective:** Maximize total net benefit in cents:
    - Sum over all items: (unit_benefit_cents[i] * q[i]) 
    - Minus sum over all items: (item_fee_cents[i] * z[i]) 
    - Minus sum over all categories: (activation_fee_cents[c] * w[c])
    - Plus sum over all bundles: (bonus_cents[bundle] * b[bundle])
7.  **Formulate Constraints:**
    -   **Item Authorization and Bounds:** For each item_ref i, q[i] = 0 if authorized[i] = 0; else q[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer).
    -   **Item Selection Indicator:** For each item_ref i, z[i] = 1 if q[i] > 0, else 0. (Enforced via q[i] ≥ minimum_lot[i] * z[i], q[i] ≤ maximum_order[i] * z[i])
    -   **Per-Area Capacity:** For each resource/area r, sum over all items assigned to r of (usage amount per unit * q[i]) ≤ total available capacity for r (sum of opening and reservation in capacity_ledger for r).
    -   **Category Quantity Bounds:** For each category c, sum over all items in c of q[i] ≥ minimum_quantity[c] * w[c], and ≤ maximum_quantity[c] * w[c]; and w[c] = 1 if any q[i] > 0 for items in c, else 0.
    -   **Incompatibility:** For each incompatible pair (i, j), z[i] + z[j] ≤ 1 (cannot select both).
    -   **Prerequisite:** For each requires pair (i, j), z[i] ≤ z[j] (if i is selected, j must also be selected).
    -   **Bundle Bonus:** For each bundle (item_a, item_b), b[bundle] ≤ z[item_a], b[bundle] ≤ z[item_b], b[bundle] ≥ z[item_a] + z[item_b] - 1 (b[bundle] = 1 iff both items are selected).
    -   **Zero for Unauthorized:** For any item_ref with authorized = 0, q[i] = 0, z[i] = 0.
    -   **Integrality:** All q[i] are integer, all z[i], w[c], b[bundle] are binary.
[Abstract Model Plan END]
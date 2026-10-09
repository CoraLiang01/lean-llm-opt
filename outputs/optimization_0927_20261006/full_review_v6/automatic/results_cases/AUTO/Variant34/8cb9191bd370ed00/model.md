[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal number of packs to display for each authorized product-and-section option (item_ref, location_id) in MARKET_SQUARE, maximizing net merchandising benefit. The plan must respect section capacities, per-item and per-category activation fees, minimum/maximum order sizes, category-wide quantity bounds, resource usage limits (with unit conversion), incompatibility and prerequisite requirements, and bundle bonuses. Only authorized options may be selected; unauthorized options must have zero quantity.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items: All item_ref values from the union of item tables (batch_01/export_07.csv and batch_02/export_08.csv), each associated with a category and location_id.
    - Categories: All category values from the category table (batch_04/export_04.csv).
    - Resources: All resource values from the capacity_ledger and usage tables (batch_03/export_03.csv, batch_06/export_12.csv, batch_01/export_13.csv).
    - Bundles: All (item_a, item_b) pairs from the bundle table (batch_02/export_02.csv).
    - Incompatibilities: All (item_a, item_b) pairs from the incompatible table (batch_06/export_06.csv).
    - Prerequisites: All (item_ref, prerequisite_ref) pairs from the requires table (batch_05/export_11.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of packs to display for item option i (item_ref, location_id). Type: GRB.INTEGER, with bounds [0, maximum_order_i], and x[i] = 0 if not authorized.
    -   `y[i]` = 1 if item option i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is selected (i.e., sum over i in c of y[i] >= 1), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected (y[item_a] = y[item_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each item option: sum of amount_cents from all benefit rows in batch_01/export_01.csv for that item_ref.
    -   Item activation fee: activation_fee_cents from item_fee table (batch_03/export_09.csv), per item_ref.
    -   Category activation fee: activation_fee_cents from category table (batch_04/export_04.csv), per category.
    -   Bundle bonus: bonus_cents from bundle table (batch_02/export_02.csv), per (item_a, item_b) pair.
    -   Minimum/maximum order: minimum_lot and maximum_order from item tables (batch_01/export_07.csv, batch_02/export_08.csv), per item_ref.
    -   Authorization: authorized field from item tables; only options with authorized=1 may be selected.
    -   Resource usage per item: amount (converted to ml if needed) from usage tables (batch_06/export_12.csv, batch_01/export_13.csv), per (item_ref, resource).
    -   Section capacity: sum of amount from capacity_ledger table (batch_03/export_03.csv), per resource (location_id), after summing all entries for each resource.
    -   Category bounds: minimum_quantity and maximum_quantity from category table (batch_04/export_04.csv), per category.
    -   Incompatibility pairs: from incompatible table (batch_06/export_06.csv).
    -   Prerequisite pairs: from requires table (batch_05/export_11.csv).
6.  **Formulate Objective:** Maximize total net merchandising benefit, defined as:
        - Sum over all item options of (per-unit benefit * x[i])
        - Minus sum over all selected item options of item activation_fee (deducted once per selected option)
        - Minus sum over all used categories of category activation_fee (deducted once per used category)
        - Plus sum over all activated bundles of bundle bonus (added once per bundle if both items are selected)
    All terms are in USD cents.
7.  **Formulate Constraints:**
    -   **Authorization:** For each item option i, x[i] = 0 if authorized_i = 0.
    -   **Order Bounds:** For each authorized item option i, minimum_lot_i * y[i] ≤ x[i] ≤ maximum_order_i * y[i]; x[i] ≥ 0 integer; y[i] ∈ {0,1}.
    -   **Section Capacity:** For each resource (section), sum over all item options assigned to that resource of (resource usage per pack in ml * x[i]) ≤ total available capacity in ml (sum of capacity_ledger entries for that resource).
    -   **Category Quantity Bounds:** For each category c, sum over all item options in c of x[i] ≥ minimum_quantity_c and ≤ maximum_quantity_c.
    -   **Category Activation:** For each category c, z[c] = 1 if any y[i] = 1 for i in c; z[c] = 0 otherwise.
    -   **Item Activation:** For each item option i, y[i] = 1 if x[i] > 0, 0 otherwise.
    -   **Bundle Activation:** For each bundle (item_a, item_b), b[bundle] = 1 if y[item_a] = y[item_b] = 1, 0 otherwise.
    -   **Incompatibility:** For each incompatible pair (item_a, item_b), y[item_a] + y[item_b] ≤ 1.
    -   **Prerequisite:** For each (item_ref, prerequisite_ref), y[item_ref] ≤ y[prerequisite_ref].
    -   **Zero for Unauthorized:** For each item option i with authorized_i = 0, enforce x[i] = 0, y[i] = 0.
    -   **Integrality:** All x[i] are integer, all y[i], z[c], b[bundle] are binary.
[Abstract Model Plan END]
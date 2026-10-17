[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal number of packs to display for each authorized product-and-section option (item_ref, location_id), maximizing net merchandising benefit. The plan must respect section capacities, per-item and per-category activation fees, minimum/maximum order sizes, resource usage (with unit conversion), category-wide quantity bounds, incompatibility and prerequisite requirements, and bundle bonuses. Unauthorized options must be set to zero.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items: All item_ref values from the union of item tables (batch_01/export_07.csv and batch_02/export_08.csv), each with associated category and location_id.
    - Categories: All category values from the category table (batch_04/export_04.csv).
    - Resources (Sections): All resource/location_id values from the capacity_ledger and usage tables.
    - Bundles: All (item_a, item_b) pairs from the bundle table (batch_02/export_02.csv).
    - Incompatibility pairs: All (item_a, item_b) from the incompatible table (batch_06/export_06.csv).
    - Prerequisite pairs: All (item_ref, prerequisite_ref) from the requires table (batch_05/export_11.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of packs to display for item option i (item_ref, location_id). Type: GRB.INTEGER, domain: 0 or [minimum_lot, maximum_order] if authorized, 0 if unauthorized.
    -   `y[i]` = 1 if item option i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is selected (i.e., sum of y[i] for i in c > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected (i.e., y[item_a] = y[item_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit: Sum of amount_cents for each item_ref from the benefit table (batch_01/export_01.csv).
    -   Item activation fee: activation_fee_cents from item_fee table (batch_03/export_09.csv), applied once per item_ref if selected.
    -   Category activation fee: activation_fee_cents from category table (batch_04/export_04.csv), applied once per category if any item in the category is selected.
    -   Bundle bonus: bonus_cents from bundle table (batch_02/export_02.csv), applied once per bundle if both items are selected.
    -   Resource usage per item: amount (converted to ml if needed) from usage tables (batch_06/export_12.csv and batch_01/export_13.csv), mapped by item_ref and resource/location_id.
    -   Section capacity: sum of amount (in ml) for each resource from capacity_ledger table (batch_03/export_03.csv).
    -   Minimum/maximum order: minimum_lot and maximum_order from item tables (batch_01/export_07.csv and batch_02/export_08.csv).
    -   Authorization: authorized field from item tables; only authorized items may be selected.
    -   Category bounds: minimum_quantity and maximum_quantity from category table (batch_04/export_04.csv).
    -   Incompatibility: item_a, item_b pairs from incompatible table (batch_06/export_06.csv).
    -   Prerequisites: item_ref, prerequisite_ref pairs from requires table (batch_05/export_11.csv).
6.  **Formulate Objective:** Maximize total net merchandising benefit, defined as:
    -   Sum over all items of (per-unit benefit * x[i])
    -   Minus sum over all selected items of item activation_fee_cents * y[i]
    -   Minus sum over all selected categories of category activation_fee_cents * z[c]
    -   Plus sum over all bundles of bonus_cents * b[bundle]
    -   (All terms in USD cents; do not convert to dollars.)
7.  **Formulate Constraints:**
    -   **Authorization and Order Bounds:** For each item option i, x[i] = 0 if not authorized; if authorized, x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i]. Enforce y[i] = 1 iff x[i] > 0.
    -   **Section Capacity:** For each resource/section, sum over all items assigned to that section of (resource usage per pack * x[i]) ≤ total available capacity (sum of capacity_ledger entries for that resource, with all quantities in ml).
    -   **Category Quantity Bounds:** For each category c, sum over all items in c of x[i] ≥ minimum_quantity[c] and ≤ maximum_quantity[c]. Enforce z[c] = 1 iff any x[i] > 0 for i in c.
    -   **Incompatibility:** For each incompatible pair (item_a, item_b), y[item_a] + y[item_b] ≤ 1 (at most one can be selected).
    -   **Prerequisites:** For each (item_ref, prerequisite_ref), y[item_ref] ≤ y[prerequisite_ref] (if item_ref is selected, prerequisite_ref must also be selected, but not vice versa).
    -   **Bundle Bonuses:** For each bundle (item_a, item_b), b[bundle] = 1 iff y[item_a] = y[item_b] = 1; b[bundle] ≤ y[item_a], b[bundle] ≤ y[item_b], b[bundle] ≥ y[item_a] + y[item_b] - 1.
    -   **Variable Domains:** All x[i] are integer, y[i], z[c], b[bundle] are binary.
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal number of packs to display for each product-and-section option (item_ref, location_id) in MARKET_SQUARE, maximizing net merchandising benefit. The plan must respect section capacities, per-option authorization and lot/order limits, resource usage, category quantity bounds, fixed activation fees, incompatibility and prerequisite requirements, and bundle bonuses.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Item options: All item_ref-location_id pairs (from the union of batch_01/export_07.csv and batch_02/export_08.csv; use all rows as the query requests direct use).
    - Categories: All unique category values from the item tables and category table.
    - Resources (sections): All unique resource/location_id values from the usage and capacity_ledger tables.
    - Bundles: All (item_a, item_b) pairs from the bundle table.
    - Incompatibilities: All (item_a, item_b) pairs from the incompatible table.
    - Prerequisites: All (item_ref, prerequisite_ref) pairs from the requires table.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of packs to display for item option i (i.e., for each item_ref-location_id). Type: GRB.INTEGER, with lower and upper bounds per option.
    -   `y[i]` = 1 if item option i is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is selected (i.e., sum of x[i] for category c > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected (i.e., both y[i_a] and y[i_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each item option: sum of amount_cents for each item_ref from benefit table (batch_01/export_01.csv).
    -   Per-option activation fee: activation_fee_cents from item_fee table (batch_03/export_09.csv).
    -   Per-category activation fee: activation_fee_cents from category table (batch_04/export_04.csv).
    -   Bundle bonuses: bonus_cents from bundle table (batch_02/export_02.csv).
    -   Resource usage per item option: amount (converted from liters to ml) from usage tables (batch_06/export_12.csv and batch_01/export_13.csv), mapped by item_ref and resource/location_id.
    -   Section (resource) capacity: sum of amount from capacity_ledger table (batch_03/export_03.csv), per resource.
    -   Per-option minimum_lot and maximum_order: from item tables (batch_01/export_07.csv and batch_02/export_08.csv).
    -   Authorization: authorized field from item tables; only options with authorized=1 may be selected.
    -   Category bounds: minimum_quantity and maximum_quantity from category table.
    -   Incompatibilities: pairs from incompatible table (batch_06/export_06.csv).
    -   Prerequisites: pairs from requires table (batch_05/export_11.csv).
6.  **Formulate Objective:** Maximize total net merchandising benefit in USD cents, defined as:
    -   Sum over all item options: (per-unit benefit * x[i]) 
    -   Minus sum over all selected item options: (item_fee for each y[i]=1)
    -   Minus sum over all used categories: (category activation_fee for each z[c]=1)
    -   Plus sum over all activated bundles: (bonus_cents for each b[bundle]=1)
7.  **Formulate Constraints:**
    -   **Authorization and Bounds:** For each item option i, x[i] = 0 if authorized=0; otherwise, minimum_lot[i] ≤ x[i] ≤ maximum_order[i], and x[i] is integer.
    -   **Section Capacity:** For each resource/section s, sum over all item options assigned to s of (resource usage per pack * x[i]) ≤ total available capacity for s (sum of capacity_ledger entries for s, after converting all units to ml).
    -   **Category Quantity Bounds:** For each category c, sum over all x[i] for options in c must be between minimum_quantity[c] and maximum_quantity[c].
    -   **Category Activation:** For each category c, z[c] = 1 if any x[i] > 0 for options in c; z[c] = 0 otherwise.
    -   **Item Activation:** For each item option i, y[i] = 1 if x[i] > 0; y[i] = 0 otherwise.
    -   **Incompatibility:** For each incompatible pair (i, j), y[i] + y[j] ≤ 1 (cannot select both).
    -   **Prerequisite:** For each (i, prereq), y[i] ≤ y[prereq] (if i is selected, prereq must also be selected).
    -   **Bundle Bonus:** For each bundle (i, j), b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1 (b[bundle]=1 iff both y[i]=1 and y[j]=1).
    -   **Zero for Unauthorized:** For any item option with authorized=0, x[i]=0 and y[i]=0.
    -   **Variable Domains:** x[i] integer, y[i], z[c], b[bundle] binary.
[Abstract Model Plan END]
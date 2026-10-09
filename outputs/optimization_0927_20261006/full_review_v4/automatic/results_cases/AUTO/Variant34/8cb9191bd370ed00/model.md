[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of packs to display for each authorized product-and-section option (item_ref), maximizing total net merchandising benefit (in USD cents). The plan must respect per-section capacity (in ml), per-category aggregate quantity bounds (with activation fees), per-item lot/order bounds (with activation fees), and logical constraints (incompatibilities, prerequisites, bundles). Only authorized options may be selected; unauthorized options must have zero quantity.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items/options: all item_ref values from the union of item tables (batch_01/export_07.csv and batch_02/export_08.csv).
    - Sections: location_id values associated with each item_ref.
    - Categories: category values from the item tables and category table.
    - Resources: resource values from usage and capacity_ledger tables.
    - Bundles: pairs (item_a, item_b) from the bundle table.
    - Incompatibilities: pairs (item_a, item_b) from the incompatible table.
    - Prerequisites: pairs (item_ref, prerequisite_ref) from the requires table.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of packs to display for item_ref i (i.e., for each authorized product-section option). Type: GRB.INTEGER, with bounds [0, maximum_order[i]] and x[i]=0 if not authorized.
    -   `y[i]` = Binary variable: 1 if item_ref i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = Binary variable: 1 if any item in category g is selected (i.e., category is active), 0 otherwise. Type: GRB.BINARY.
    -   `b[p]` = Binary variable: 1 if both items in bundle pair p are selected (i.e., both x[item_a] > 0 and x[item_b] > 0), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each item_ref: sum of amount_cents from all benefit table rows for that item_ref (batch_01/export_01.csv).
    -   Per-item activation fee: activation_fee_cents from item_fee table (batch_03/export_09.csv).
    -   Per-category minimum/maximum quantity and activation fee: minimum_quantity, maximum_quantity, activation_fee_cents from category table (batch_04/export_04.csv).
    -   Per-item minimum_lot and maximum_order: from item tables (batch_01/export_07.csv and batch_02/export_08.csv).
    -   Authorization: authorized field from item tables; only options with authorized=1 are eligible.
    -   Resource usage per item: amount (converted from liters to ml) from usage tables (batch_06/export_12.csv and batch_01/export_13.csv), by item_ref and resource.
    -   Section capacity: sum of amount (in ml) from capacity_ledger table (batch_03/export_03.csv), by resource (section).
    -   Incompatibilities: item_a, item_b pairs from incompatible table (batch_06/export_06.csv).
    -   Prerequisites: item_ref, prerequisite_ref pairs from requires table (batch_05/export_11.csv).
    -   Bundles: item_a, item_b, bonus_cents from bundle table (batch_02/export_02.csv).
    -   Category membership: category field in item tables.
    -   Section membership: location_id field in item tables.
6.  **Formulate Objective:** Maximize total net merchandising benefit in USD cents, calculated as:
    -   Sum over all item_refs of (per-unit benefit * x[i])
    -   Minus sum over all item_refs of (item_fee[i] * y[i]) [item activation fee, once per selected item]
    -   Minus sum over all categories of (category activation_fee_cents * z[g]) [category activation fee, once per active category]
    -   Plus sum over all bundle pairs of (bonus_cents * b[p]) [bundle bonus, once per pair if both items selected]
7.  **Formulate Constraints:**
    -   **Authorization and Bounds:** For each item_ref, x[i] = 0 if authorized[i] = 0; for authorized options, minimum_lot[i] * y[i] ≤ x[i] ≤ maximum_order[i] * y[i]; y[i] ∈ {0,1}.
    -   **Section Capacity:** For each section/resource s, sum over all item_refs assigned to s of (usage[i,s] * x[i]) ≤ total available capacity for s (sum of capacity_ledger amounts for s, all in ml).
    -   **Category Quantity Bounds:** For each category g, minimum_quantity[g] * z[g] ≤ sum over i in g of x[i] ≤ maximum_quantity[g] * z[g]; z[g] ∈ {0,1}; for each i in g, y[i] ≤ z[g].
    -   **Incompatibility:** For each incompatible pair (i,j), y[i] + y[j] ≤ 1.
    -   **Prerequisite:** For each (i, prereq), y[i] ≤ y[prereq].
    -   **Bundle:** For each bundle pair (i,j), b[p] ≤ y[i], b[p] ≤ y[j], b[p] ≥ y[i] + y[j] - 1; b[p] ∈ {0,1}.
    -   **Variable Domains:** x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer), y[i] ∈ {0,1}, z[g] ∈ {0,1}, b[p] ∈ {0,1}.
[Abstract Model Plan END]
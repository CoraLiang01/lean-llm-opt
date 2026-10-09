[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of packs to display for each authorized product-and-section option (item_ref), maximizing total net merchandising benefit (in USD cents), subject to section capacity, per-item and per-category activation fees, minimum/maximum order sizes, resource usage, category quantity bounds, incompatibility and prerequisite requirements, and bundle bonuses.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items/options: all item_ref values from the union of item tables (batch_01/export_07.csv and batch_02/export_08.csv).
    - Categories: all category values from the category table (batch_04/export_04.csv).
    - Resources/sections: all resource/location_id values from the capacity_ledger and usage tables.
    - Bundles: all (item_a, item_b) pairs from the bundle table (batch_02/export_02.csv).
    - Incompatibility pairs: all (item_a, item_b) from the incompatible table (batch_06/export_06.csv).
    - Prerequisite pairs: all (item_ref, prerequisite_ref) from the requires table (batch_05/export_11.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = integer number of packs to display for item_ref i (0 if unauthorized; otherwise, 0 or integer in [minimum_lot, maximum_order]). Type: GRB.INTEGER.
    -   `y[i]` = binary indicator: 1 if item_ref i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = binary indicator: 1 if any item in category g is selected, 0 otherwise. Type: GRB.BINARY.
    -   `b[a,b]` = binary indicator: 1 if both item_a and item_b in bundle (a,b) are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each item_ref: sum of amount_cents from all benefit table rows with matching item_ref (batch_01/export_01.csv).
    -   Item activation fee: activation_fee_cents from item_fee table (batch_03/export_09.csv), per item_ref.
    -   Category activation fee: activation_fee_cents from category table (batch_04/export_04.csv), per category.
    -   Bundle bonus: bonus_cents from bundle table (batch_02/export_02.csv), per (item_a, item_b) pair.
    -   Minimum/maximum order: minimum_lot, maximum_order from item tables (batch_01/export_07.csv and batch_02/export_08.csv), per item_ref.
    -   Authorization: authorized from item tables, per item_ref.
    -   Category membership: category from item tables, per item_ref.
    -   Section/resource assignment: location_id from item tables, per item_ref.
    -   Resource usage per pack: amount (converted to ml if needed) from usage tables (batch_06/export_12.csv and batch_01/export_13.csv), per (item_ref, resource).
    -   Section/resource capacity: sum of amount from capacity_ledger table (batch_03/export_03.csv), per resource (convert all to ml).
    -   Incompatibility: pairs from incompatible table (batch_06/export_06.csv).
    -   Prerequisite: pairs from requires table (batch_05/export_11.csv).
    -   Category quantity bounds: minimum_quantity, maximum_quantity from category table (batch_04/export_04.csv), per category.
6.  **Formulate Objective:** Maximize total net merchandising benefit, defined as:
    -   Sum over all item_ref of (per-unit benefit * x[i])
    -   Minus sum over all item_ref of (item activation fee * y[i])
    -   Minus sum over all categories of (category activation fee * z[g])
    -   Plus sum over all bundles of (bundle bonus * b[a,b])
    All terms are in USD cents.
7.  **Formulate Constraints:**
    -   **Authorization and Order Bounds:** For each item_ref, x[i] = 0 if not authorized; if authorized, x[i] ∈ {0} ∪ [minimum_lot, maximum_order], integer.
    -   **Item Activation Linking:** For each item_ref, y[i] = 1 if x[i] > 0, y[i] = 0 if x[i] = 0; enforce with x[i] ≥ minimum_lot * y[i], x[i] ≤ maximum_order * y[i].
    -   **Category Activation Linking:** For each category g, z[g] = 1 if any y[i] = 1 for item_ref in g; enforce y[i] ≤ z[g] for all i in g, z[g] ≤ sum(y[i] for i in g).
    -   **Category Quantity Bounds:** For each category g, sum(x[i] for i in g) ≥ minimum_quantity[g], sum(x[i] for i in g) ≤ maximum_quantity[g].
    -   **Section/Resource Capacity:** For each resource/section r, sum over all item_ref assigned to r of (resource usage per pack * x[i]) ≤ total available capacity for r (all in ml).
    -   **Incompatibility:** For each incompatible pair (a,b), y[a] + y[b] ≤ 1.
    -   **Prerequisite:** For each (item_ref, prerequisite_ref), y[item_ref] ≤ y[prerequisite_ref].
    -   **Bundle Linking:** For each bundle (a,b), b[a,b] ≤ y[a], b[a,b] ≤ y[b], b[a,b] ≥ y[a] + y[b] - 1.
[Abstract Model Plan END]
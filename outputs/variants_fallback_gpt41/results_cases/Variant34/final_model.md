[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of packs to display for each authorized product-and-section option (item_ref, location_id), maximizing total net merchandising benefit (sum of per-unit benefits, minus fixed item and category activation fees, plus bundle bonuses), subject to section capacity, category quantity bounds, resource usage, incompatibility, and prerequisite requirements. Only authorized options may be selected, and each section’s capacity is independent.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items: All item_ref values from the union of both item tables (batch_01/export_07.csv and batch_02/export_08.csv), each with associated category and location_id.
    - Categories: All category values from the category table (batch_04/export_04.csv).
    - Resources: All resource values from the capacity_ledger and usage tables (batch_03/export_03.csv, batch_06/export_12.csv, batch_01/export_13.csv).
    - Bundles: All (item_a, item_b) pairs from the bundle table (batch_02/export_02.csv).
    - Incompatibility pairs: All (item_a, item_b) from the incompatible table (batch_06/export_06.csv).
    - Prerequisite pairs: All (item_ref, prerequisite_ref) from the requires table (batch_05/export_11.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of packs to display for item option i (item_ref, location_id). Type: GRB.INTEGER, with bounds [0, maximum_order_i] and x[i]=0 if not authorized.
    -   `y[i]` = Binary variable: 1 if item option i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = Binary variable: 1 if any item in category c is selected (i.e., sum over i in c of y[i] ≥ 1), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = Binary variable: 1 if both items in bundle are selected (y[item_a]=1 and y[item_b]=1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each item option: sum of amount_cents from all benefit rows for that item_ref (batch_01/export_01.csv).
    -   Item activation fee: activation_fee_cents from item_fee table (batch_03/export_09.csv), applied once per item option with x[i]>0.
    -   Category activation fee: activation_fee_cents from category table (batch_04/export_04.csv), applied once per category if any item in that category is selected.
    -   Bundle bonus: bonus_cents from bundle table (batch_02/export_02.csv), applied once per bundle if both items are selected.
    -   Section capacity: For each resource (section), sum amount from capacity_ledger table (batch_03/export_03.csv) for that resource (convert all to ml).
    -   Item resource usage: For each item option, amount from usage tables (batch_06/export_12.csv and batch_01/export_13.csv), convert all to ml (1 liter = 1000 ml).
    -   Category quantity bounds: minimum_quantity and maximum_quantity from category table (batch_04/export_04.csv).
    -   Authorization, minimum_lot, maximum_order, category, location_id: from item tables (batch_01/export_07.csv and batch_02/export_08.csv).
    -   Incompatibility and prerequisite pairs: from incompatible (batch_06/export_06.csv) and requires (batch_05/export_11.csv) tables.
6.  **Formulate Objective:** Maximize total net merchandising benefit:
    -   Sum over all item options of (per-unit benefit × x[i])
    -   Minus sum over all selected item options of item activation_fee (applied once per selected item)
    -   Minus sum over all activated categories of category activation_fee (applied once per activated category)
    -   Plus sum over all bundles of bundle bonus (applied once per bundle if both items are selected)
    -   All terms in USD cents.
7.  **Formulate Constraints:**
    -   **Authorization and Bounds:** For each item option i, x[i]=0 if not authorized; otherwise, minimum_lot_i × y[i] ≤ x[i] ≤ maximum_order_i × y[i], with y[i] ∈ {0,1}.
    -   **Section Capacity:** For each resource (section), sum over all item options assigned to that section of (resource usage per pack × x[i]) ≤ total available capacity for that section (in ml).
    -   **Category Quantity Bounds:** For each category c, sum over all item options in c of x[i] ≥ minimum_quantity_c and ≤ maximum_quantity_c (unconditional, even if no items are selected).
    -   **Category Activation:** For each category c, z[c] = 1 if any y[i]=1 for i in c; z[c]=0 otherwise.
    -   **Item Activation:** For each item option i, y[i]=1 if x[i]>0; y[i]=0 if x[i]=0.
    -   **Bundle Activation:** For each bundle (item_a, item_b), b[bundle]=1 if y[item_a]=1 and y[item_b]=1; b[bundle]=0 otherwise.
    -   **Incompatibility:** For each incompatible pair (item_a, item_b), y[item_a] + y[item_b] ≤ 1.
    -   **Prerequisite:** For each (item_ref, prerequisite_ref), y[item_ref] ≤ y[prerequisite_ref] (i.e., cannot select item_ref unless prerequisite_ref is also selected).
    -   **Variable Domains:** x[i] ∈ {0} ∪ [minimum_lot_i, maximum_order_i] (integer), y[i], z[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]
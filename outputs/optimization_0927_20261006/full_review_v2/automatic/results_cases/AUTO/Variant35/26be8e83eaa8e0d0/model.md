[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate licensed game-edition packages (item_ref) to PC, CONSOLE, and MOBILE platforms, maximizing net licensing return (benefit after all fees and bonuses), subject to platform-specific memory capacity, per-item and per-category quantity bounds, authorization, incompatibility, prerequisite, and bundle bonus rules. All data is to be taken directly from the supplied tables.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items: set of all item_ref from the item table (export_08.csv).
    - Platforms: {PC, CONSOLE, MOBILE}, from location_id/resource fields.
    - Categories: set of all category from the category table (export_04.csv).
    - Bundles: set of (item_a, item_b) pairs from the bundle table (export_02.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = integer quantity of item i (item_ref) to allocate. Type: GRB.INTEGER. Domain: {0} if unauthorized; else {0} ∪ [minimum_lot, maximum_order].
    -   `z[i]` = binary variable: 1 if item i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[c]` = binary variable: 1 if any item in category c is selected (sum over i in c of z[i] ≥ 1), 0 otherwise. Type: GRB.BINARY.
    -   `b[a,b]` = binary variable: 1 if both items a and b in a bundle are selected (z[a]=1 and z[b]=1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each item: sum of all benefit components for item_ref, each converted to USD cents using fx table (amount * usd_cents_numerator / denominator).
    -   Item activation fee: activation_fee_cents from item_fee table (export_09.csv), per item_ref.
    -   Category activation fee: activation_fee_cents from category table (export_04.csv), per category.
    -   Bundle bonus: bonus_cents from bundle table (export_02.csv), per (item_a, item_b) pair.
    -   Memory usage per unit: amount (in GB) from usage table (export_12.csv), converted to MB (×1000), per item_ref and resource.
    -   Platform memory capacity: sum of capacity_ledger entries (opening + reservation) in MB, per resource (PC, CONSOLE, MOBILE).
    -   Authorization, minimum_lot, maximum_order, category, and platform: from item table (export_08.csv).
    -   Incompatibility: pairs (item_a, item_b) from incompatible table (export_07.csv).
    -   Prerequisite: pairs (item_ref, prerequisite_ref) from requires table (export_11.csv).
    -   Category quantity bounds: minimum_quantity, maximum_quantity from category table (export_04.csv).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (per-unit benefit in USD cents) × x[i]
    -   Minus: sum over all items with x[i]>0 of item activation_fee_cents
    -   Minus: sum over all categories with any item selected of category activation_fee_cents
    -   Plus: sum over all bundles where both items are selected of bonus_cents
7.  **Formulate Constraints:**
    -   Platform Memory Capacity: For each platform/resource, sum over all items assigned to that platform of (memory usage per unit in MB × x[i]) ≤ total available MB for that platform.
    -   Authorization and Quantity Bounds: For each item, x[i] = 0 if authorized=0; else x[i] ∈ {0} ∪ [minimum_lot, maximum_order] (integer).
    -   Item Activation Linking: For each item, z[i]=1 if x[i]>0, z[i]=0 if x[i]=0; enforce x[i] ≤ maximum_order × z[i] and x[i] ≥ minimum_lot × z[i] for authorized items.
    -   Category Quantity Bounds: For each category, sum over all items in that category of x[i] ≥ minimum_quantity and ≤ maximum_quantity.
    -   Category Activation Linking: For each category, w[c]=1 if any item in c is selected (sum z[i] over i in c ≥ 1), w[c]=0 otherwise.
    -   Incompatibility: For each incompatible pair (a,b), z[a] + z[b] ≤ 1.
    -   Prerequisite: For each (item_ref, prerequisite_ref), z[item_ref] ≤ z[prerequisite_ref].
    -   Bundle Bonus Linking: For each bundle (a,b), b[a,b] ≤ z[a], b[a,b] ≤ z[b], b[a,b] ≥ z[a] + z[b] - 1.
[Abstract Model Plan END]
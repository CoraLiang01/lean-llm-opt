[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate licensed game-edition packages (item_ref) to PC, CONSOLE, and MOBILE platforms, maximizing net licensing return (benefit after fees and bonuses), subject to per-platform memory capacity, per-category quantity bounds, item and category activation fees, incompatibility and prerequisite constraints, bundle bonuses, and only allowing authorized options. All data is to be taken directly from the supplied tables.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, requires, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (I): All item_ref from the item table (export_08.csv).
    - Platforms (P): PC, CONSOLE, MOBILE (from location_id/resource fields).
    - Categories (C): All category from the category table (export_04.csv).
    - Bundles (B): All bundle pairs (item_a, item_b) from the bundle table (export_02.csv).
    - Incompatibilities (INC): All (item_a, item_b) pairs from the incompatible table (export_07.csv).
    - Prerequisites (REQ): All (item_ref, prerequisite_ref) pairs from the requires table (export_11.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item i to allocate (must be 0 if unauthorized; otherwise, 0 or an integer between minimum_lot and maximum_order for item i). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is selected (sum over i in c of y[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle b are selected (y[item_a] = y[item_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit per unit for each item: computed from benefit table (export_01.csv), using amount * usd_cents_numerator / denominator (from fx table, export_05.csv), summed over all components for each item_ref.
    -   Item activation fee: activation_fee_cents from item_fee table (export_09.csv), applied once per item if y[i]=1.
    -   Bundle bonus: bonus_cents from bundle table (export_02.csv), applied once per bundle if both items are selected.
    -   Category bounds and activation fee: minimum_quantity, maximum_quantity, activation_fee_cents from category table (export_04.csv).
    -   Item authorization, minimum_lot, maximum_order, category, and platform: from item table (export_08.csv).
    -   Per-item resource usage: amount (in GB, to be converted to MB) from usage table (export_12.csv), mapped by item_ref and resource.
    -   Platform memory capacity: sum of amount from capacity_ledger table (export_03.csv) for each resource (platform).
    -   Incompatibility and prerequisite relations: from incompatible (export_07.csv) and requires (export_11.csv) tables.
6.  **Formulate Objective:** Maximize total net benefit in USD cents, defined as:
    -   Sum over all items: (total per-unit benefit in USD cents) × x[i]
    -   Minus: sum over all items: item activation_fee_cents × y[i]
    -   Minus: sum over all categories: category activation_fee_cents × z[c]
    -   Plus: sum over all bundles: bonus_cents × w[b]
7.  **Formulate Constraints:**
    -   **Authorization and Quantity Bounds:** For each item i, if authorized=0 then x[i]=0; if authorized=1, then x[i]=0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i] (integer).
    -   **Item Activation Linking:** For each item i, y[i]=1 if x[i]>0, y[i]=0 if x[i]=0.
    -   **Category Activation Linking:** For each category c, z[c]=1 if any y[i]=1 for items in c; z[c]=0 otherwise.
    -   **Category Quantity Bounds:** For each category c, sum over i in c of x[i] ≥ minimum_quantity[c] and ≤ maximum_quantity[c].
    -   **Platform Memory Capacity:** For each platform p, sum over items i assigned to p of (usage[i,p] in MB) × x[i] ≤ total available capacity for p (sum of opening and reservation entries in capacity_ledger for p).
    -   **Incompatibility:** For each (item_a, item_b) in INC, y[item_a] + y[item_b] ≤ 1.
    -   **Prerequisite:** For each (item_ref, prerequisite_ref) in REQ, y[item_ref] ≤ y[prerequisite_ref].
    -   **Bundle Bonus Linking:** For each bundle b = (item_a, item_b), w[b] ≤ y[item_a], w[b] ≤ y[item_b], w[b] ≥ y[item_a] + y[item_b] - 1.
[Abstract Model Plan END]
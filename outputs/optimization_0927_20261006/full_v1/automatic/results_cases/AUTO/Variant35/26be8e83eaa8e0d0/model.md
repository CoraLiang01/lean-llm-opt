[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate licensed game-edition packages (item_ref) to PC, CONSOLE, and MOBILE platforms, maximizing net licensing return (benefit after all fees and bonuses), subject to platform-specific memory capacity, item/category quantity bounds, authorization, incompatibility, prerequisite, and bundle bonus rules. All data is to be taken directly from the supplied tables.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items: set of all item_ref from the item table (export_08.csv).
    - Platforms: {PC, CONSOLE, MOBILE}, from location_id/resource fields.
    - Categories: set of all category from the category table (export_04.csv).
    - Bundles: set of (item_a, item_b) pairs from the bundle table (export_02.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = integer quantity of item_ref i to allocate (0 if unauthorized; otherwise, 0 or any integer between minimum_lot and maximum_order for i). Type: GRB.INTEGER.
    -   `z[i]` = binary variable: 1 if item_ref i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[c]` = binary variable: 1 if any item in category c is selected (sum over i in c of z[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[a,b]` = binary variable: 1 if both items a and b in a bundle are selected (z[a]=1 and z[b]=1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit per unit for each item: computed from benefit table (export_01.csv), using amount * usd_cents_numerator / denominator (from fx table, export_05.csv) for each component, summed per item_ref.
    -   Item activation fee: from item_fee table (export_09.csv), field activation_fee_cents.
    -   Category activation fee and quantity bounds: from category table (export_04.csv), fields activation_fee_cents, minimum_quantity, maximum_quantity.
    -   Bundle bonus: from bundle table (export_02.csv), field bonus_cents.
    -   Memory usage per unit: from usage table (export_12.csv), field amount (convert GB to MB using 1000 MB/GB), per item_ref and resource.
    -   Platform memory capacity: from capacity_ledger table (export_03.csv), sum of amount per resource (MB).
    -   Authorization, minimum_lot, maximum_order, category, and platform assignment: from item table (export_08.csv).
    -   Incompatibility pairs: from incompatible table (export_07.csv).
    -   Prerequisite pairs: from requires table (export_11.csv).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (total benefit per unit * x[i]) 
    -   Minus sum over all selected items: item activation_fee_cents * z[i]
    -   Minus sum over all used categories: category activation_fee_cents * w[c]
    -   Plus sum over all selected bundles: bonus_cents * b[a,b]
7.  **Formulate Constraints:**
    -   **Authorization and Quantity Bounds:** For each item i, x[i] = 0 if authorized=0; if authorized=1, x[i] ∈ {0} ∪ [minimum_lot, maximum_order] (integer).
    -   **Item Activation Linking:** For each item i, z[i]=1 if x[i]>0, z[i]=0 if x[i]=0; enforce with x[i] ≤ maximum_order[i] * z[i] and x[i] ≥ minimum_lot[i] * z[i] for authorized items.
    -   **Platform Memory Capacity:** For each platform p, sum over items assigned to p of (usage per unit in MB * x[i]) ≤ total available MB for p (from capacity_ledger).
    -   **Category Quantity Bounds:** For each category c, sum over items in c of x[i] ≥ minimum_quantity[c] and ≤ maximum_quantity[c].
    -   **Category Activation Linking:** For each category c, w[c]=1 if any z[i]=1 for i in c; w[c]=0 otherwise; enforce with z[i] ≤ w[c] for all i in c.
    -   **Incompatibility:** For each incompatible pair (i,j), z[i] + z[j] ≤ 1.
    -   **Prerequisite:** For each (i, prereq), z[i] ≤ z[prereq].
    -   **Bundle Bonus Linking:** For each bundle (a,b), b[a,b] ≤ z[a], b[a,b] ≤ z[b], b[a,b] ≥ z[a] + z[b] - 1.
[Abstract Model Plan END]
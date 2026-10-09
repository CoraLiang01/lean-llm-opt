[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate licensed game-edition packages (item_ref) to PC, CONSOLE, and MOBILE platforms, maximizing net licensing return (benefit after all fees and bonuses), subject to platform-specific memory (capacity) limits, per-item and per-category constraints, incompatibility and prerequisite rules, and bundle bonuses. Only authorized options may be chosen, and all quantities must be integer multiples within allowed bounds.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (item_ref): Each edition-platform option (from the item table, all 30 rows).
    - Platforms (location_id): PC, CONSOLE, MOBILE (from item table and usage/capacity_ledger).
    - Categories (category): G0, G1, G2, G3 (from category table).
    - Bundles: Pairs of items eligible for a bonus (from bundle table).
    - Incompatibility pairs: Pairs of items that cannot be jointly selected (from incompatible table).
    - Prerequisite pairs: (item_ref, prerequisite_ref) pairs where one requires the other (from requires table).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item_ref i to allocate (must be 0 if unauthorized, else 0 or between minimum_lot and maximum_order). Type: GRB.INTEGER.
    -   `z[i]` = Binary variable: 1 if item_ref i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[c]` = Binary variable: 1 if any item in category c is selected (category activation fee applies), 0 otherwise. Type: GRB.BINARY.
    -   `b[pair]` = Binary variable: 1 if both items in a bundle pair are selected (bonus applies), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit per unit for each item: Calculated by summing all benefit components for item_ref i (from benefit table), converting each amount to USD cents using fx table (amount * usd_cents_numerator / denominator), and summing signed values.
    -   Item activation fee: activation_fee_cents per item_ref (from item_fee table).
    -   Category activation fee: activation_fee_cents per category (from category table).
    -   Bundle bonus: bonus_cents per bundle pair (from bundle table).
    -   Memory usage per unit: amount (in GB, convert to MB) per item_ref and platform (from usage table).
    -   Platform memory capacity: sum of capacity_ledger entries per resource (from capacity_ledger table).
    -   Minimum/maximum order per item: minimum_lot, maximum_order (from item table).
    -   Authorization: authorized (from item table; only items with authorized=1 may be chosen).
    -   Category membership: category per item_ref (from item table).
    -   Platform assignment: location_id per item_ref (from item table).
    -   Incompatibility pairs: item_a, item_b (from incompatible table).
    -   Prerequisite pairs: item_ref, prerequisite_ref (from requires table).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (benefit per unit * x[i])
    -   Minus: sum of item activation fees for each item selected (activation_fee_cents * z[i])
    -   Minus: sum of category activation fees for each category used (activation_fee_cents * w[c])
    -   Plus: sum of bundle bonuses for each bundle pair where both items are selected (bonus_cents * b[pair])
7.  **Formulate Constraints:**
    -   **Authorization:** For each item_ref i, if authorized=0, x[i]=0.
    -   **Item Quantity Bounds:** For each authorized item_ref i, x[i]=0 or minimum_lot ≤ x[i] ≤ maximum_order.
    -   **Item Activation Linking:** For each item_ref i, z[i]=1 if x[i]>0, else z[i]=0. (Enforced via x[i] ≥ minimum_lot * z[i], x[i] ≤ maximum_order * z[i])
    -   **Category Quantity Bounds:** For each category c, sum of x[i] over all items in c must be between minimum_quantity and maximum_quantity (from category table).
    -   **Category Activation Linking:** For each category c, w[c]=1 if any x[i]>0 for i in c, else w[c]=0.
    -   **Platform Memory Capacity:** For each platform p (PC, CONSOLE, MOBILE), sum over all items assigned to p of (x[i] * usage per unit in MB) ≤ total available MB for p (from capacity_ledger).
    -   **Incompatibility:** For each incompatible pair (i, j), z[i] + z[j] ≤ 1 (cannot both be selected).
    -   **Prerequisite:** For each (i, prereq), z[i] ≤ z[prereq] (if i is selected, prereq must also be selected).
    -   **Bundle Bonus Linking:** For each bundle pair (i, j), b[pair] ≤ z[i], b[pair] ≤ z[j], b[pair] ≥ z[i] + z[j] - 1 (b[pair]=1 iff both z[i]=z[j]=1).
    -   **Variable Domains:** x[i] ∈ {0} ∪ [minimum_lot, maximum_order] (integer), z[i], w[c], b[pair] ∈ {0,1}.
[Abstract Model Plan END]
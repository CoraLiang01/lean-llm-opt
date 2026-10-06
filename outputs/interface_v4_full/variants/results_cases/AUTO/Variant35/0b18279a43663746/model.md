[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate licensed game-edition packages (item/platform options) to PC, CONSOLE, and MOBILE, maximizing net licensing return (benefit after fees and bonuses), subject to per-platform memory capacity, per-category quantity bounds, item/category activation fees, incompatibility and prerequisite rules, bundle bonuses, and only allowing authorized options. All data is to be taken directly from the supplied tables.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items/options: all rows in export_08.csv (item_ref, each with category and platform/location_id)
    - Categories: all rows in export_04.csv (category)
    - Platforms/resources: PC, CONSOLE, MOBILE (from location_id/resource in usage/capacity_ledger)
    - Bundles: all rows in export_02.csv (pairs of item_refs)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item_ref i to allocate (must be 0 if unauthorized; otherwise, 0 or between minimum_lot and maximum_order). Type: GRB.INTEGER.
    -   `z[i]` = Binary variable: 1 if item_ref i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[c]` = Binary variable: 1 if any item in category c is selected (sum over i in c of x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = Binary variable: 1 if both items in bundle are selected (for each bundle row), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit per unit: For each item_ref, sum all benefit components (from export_01.csv) after converting amount * usd_cents_numerator / denominator using fx (export_05.csv), to get benefit in USD cents per unit.
    -   Item activation fee: activation_fee_cents per item_ref from export_09.csv.
    -   Category activation fee: activation_fee_cents per category from export_04.csv.
    -   Bundle bonus: bonus_cents per bundle from export_02.csv.
    -   Memory usage: amount (in GB, convert to MB) per item_ref/platform from export_12.csv.
    -   Platform memory capacity: sum of capacity_ledger entries per resource (export_03.csv).
    -   Category bounds: minimum_quantity, maximum_quantity per category from export_04.csv.
    -   Authorization, minimum_lot, maximum_order: per item_ref from export_08.csv.
    -   Incompatibility: pairs of item_refs from export_07.csv.
    -   Prerequisite: pairs of item_refs from export_11.csv.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (benefit per unit) * x[i]
    -   Minus: sum over all items: item activation_fee_cents * z[i] (charged once per item if any quantity is selected)
    -   Minus: sum over all categories: category activation_fee_cents * w[c] (charged once per category if any item in category is selected)
    -   Plus: sum over all bundles: bonus_cents * b[bundle] (only if both items in bundle are selected)
7.  **Formulate Constraints:**
    -   **Authorization:** For each item_ref, if authorized == 0, x[i] = 0.
    -   **Item quantity bounds:** For each authorized item_ref, x[i] = 0 or minimum_lot ≤ x[i] ≤ maximum_order.
    -   **Item activation linking:** For each item_ref, z[i] = 1 if x[i] > 0, else 0. (Enforced via x[i] ≤ maximum_order * z[i], x[i] ≥ minimum_lot * z[i] for authorized items.)
    -   **Category activation linking:** For each category c, w[c] = 1 if any x[i] > 0 for i in c, else 0. (Enforced via sum_{i in c} x[i] ≥ minimum_lot * w[c], sum_{i in c} x[i] ≤ (sum of maximum_order in c) * w[c])
    -   **Category quantity bounds:** For each category c, sum_{i in c} x[i] ≥ minimum_quantity, sum_{i in c} x[i] ≤ maximum_quantity.
    -   **Platform memory capacity:** For each platform/resource r, sum over items assigned to r of (usage in MB per unit) * x[i] ≤ total available MB for r (sum of capacity_ledger entries for r).
    -   **Incompatibility:** For each incompatible pair (i, j), z[i] + z[j] ≤ 1.
    -   **Prerequisite:** For each (i, prereq), z[i] ≤ z[prereq].
    -   **Bundle bonus linking:** For each bundle (i, j), b[bundle] ≤ z[i], b[bundle] ≤ z[j], b[bundle] ≥ z[i] + z[j] - 1.
    -   **Variable domains:** x[i] ∈ {0} ∪ [minimum_lot, maximum_order] (integer), z[i], w[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]
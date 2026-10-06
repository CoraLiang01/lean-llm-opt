[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate licensed game-edition packages (item/platform options) to PC, CONSOLE, and MOBILE, maximizing net licensing return (benefit after all fees and bonuses), subject to platform-specific memory capacity, per-category quantity bounds, item/category activation fees, incompatibility and prerequisite rules, bundle bonuses, and only using authorized options. All data is to be taken directly from the supplied tables.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, and logical (incompatibility, prerequisite) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items/options: All rows in export_08.csv (item_ref, each with category and platform/location_id).
    - Platforms: {PC, CONSOLE, MOBILE} (from location_id/resource).
    - Categories: All rows in export_04.csv (category).
    - Bundles: All rows in export_02.csv (pairs of item_refs).
    - Incompatibility pairs: All rows in export_07.csv (item_a, item_b).
    - Prerequisite pairs: All rows in export_11.csv (item_ref, prerequisite_ref).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item_ref i to allocate (must be 0 if unauthorized; otherwise, 0 or between minimum_lot and maximum_order for that item). Type: GRB.INTEGER.
    -   `z[i]` = Binary variable: 1 if item_ref i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[c]` = Binary variable: 1 if any item in category c is selected (sum over i in c of z[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = Binary variable: 1 if both items in bundle are selected (z[item_a] = z[item_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit per unit for each item: Calculated from export_01.csv (sum of all 'amount' for each item_ref, converted to USD cents using export_05.csv fx rates).
    -   Item activation fee: export_09.csv (activation_fee_cents per item_ref).
    -   Category activation fee: export_04.csv (activation_fee_cents per category).
    -   Bundle bonus: export_02.csv (bonus_cents per bundle).
    -   Memory usage per unit: export_12.csv (amount in GB per item_ref/resource, convert to MB).
    -   Platform memory capacity: export_03.csv (sum of 'amount' for each resource, in MB).
    -   Category quantity bounds: export_04.csv (minimum_quantity, maximum_quantity per category).
    -   Authorization, min/max order: export_08.csv (authorized, minimum_lot, maximum_order per item_ref).
    -   Incompatibility: export_07.csv (item_a, item_b pairs).
    -   Prerequisites: export_11.csv (item_ref, prerequisite_ref pairs).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (benefit_per_unit[i] * x[i]) 
    -   Minus sum over all selected items: (item activation_fee_cents[i] * z[i])
    -   Minus sum over all used categories: (category activation_fee_cents[c] * w[c])
    -   Plus sum over all bundles: (bonus_cents[bundle] * b[bundle])
7.  **Formulate Constraints:**
    -   **Authorization:** For each item_ref i, if authorized = 0, x[i] = 0.
    -   **Item quantity bounds:** For each authorized item_ref i, x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i].
    -   **Item activation linking:** For each item_ref i, z[i] = 1 if x[i] > 0, else 0. (Enforced via x[i] ≤ maximum_order[i] * z[i], x[i] ≥ minimum_lot[i] * z[i] for authorized items.)
    -   **Category activation linking:** For each category c, w[c] = 1 if any z[i] for i in c is 1, else 0. (Enforced via z[i] ≤ w[c] for all i in c; w[c] ≥ max(z[i] for i in c).)
    -   **Category quantity bounds:** For each category c, sum over i in c of x[i] ≥ minimum_quantity[c], and ≤ maximum_quantity[c].
    -   **Platform memory capacity:** For each platform p (PC, CONSOLE, MOBILE), sum over all items i assigned to p of (usage_per_unit[i,p] * x[i]) ≤ total_capacity[p] (in MB).
    -   **Incompatibility:** For each incompatible pair (i, j), z[i] + z[j] ≤ 1.
    -   **Prerequisite:** For each (i, prereq), z[i] ≤ z[prereq].
    -   **Bundle bonus linking:** For each bundle (item_a, item_b), b[bundle] ≤ z[item_a], b[bundle] ≤ z[item_b], b[bundle] ≥ z[item_a] + z[item_b] - 1.
    -   **Variable domains:** x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer), z[i], w[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]
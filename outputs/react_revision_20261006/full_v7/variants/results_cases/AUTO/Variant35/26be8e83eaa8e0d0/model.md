[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate licensed game-edition packages (item/platform options) to PC, CONSOLE, and MOBILE, maximizing net licensing return (benefit after all fees and bonuses), subject to per-platform memory capacity, per-category quantity bounds, item authorization, minimum/maximum order sizes, incompatibility and prerequisite rules, and bundle bonuses. All data is to be taken directly from the supplied tables; only authorized options may be chosen, and all quantities must be integer multiples within allowed bounds.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items/options (`I`): Each row in the item table (export_08.csv), uniquely identified by `item_ref`.
    - Platforms (`P`): PC, CONSOLE, MOBILE (from `location_id` in item table and `resource` in usage/capacity tables).
    - Categories (`C`): Each unique `category` in the category table (export_04.csv).
    - Bundles (`B`): Each row in the bundle table (export_02.csv), with pairs of items.
    - Incompatibility pairs and prerequisite pairs (from incompatible and requires tables).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item option `i` to allocate (must be 0 if unauthorized; otherwise, 0 or between minimum_lot and maximum_order). Type: GRB.INTEGER.
    -   `z[i]` = Binary variable: 1 if item option `i` is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `y[c]` = Binary variable: 1 if any item in category `c` is selected (i.e., sum of x[i] for items in c > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = Binary variable: 1 if both items in bundle `bundle` are selected (i.e., both x[item_a] > 0 and x[item_b] > 0), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit per unit for each item: Calculated from export_01.csv (sum of all benefit components for each item, converted to USD cents using fx table export_05.csv).
    -   Item activation fee: From export_09.csv (`activation_fee_cents` per item).
    -   Bundle bonus: From export_02.csv (`bonus_cents` per bundle).
    -   Category bounds and activation fee: From export_04.csv (`minimum_quantity`, `maximum_quantity`, `activation_fee_cents` per category).
    -   Item authorization, minimum_lot, maximum_order, category, and platform: From export_08.csv.
    -   Per-item memory usage: From export_12.csv (`amount` in GB, convert to MB).
    -   Per-platform available memory: From export_03.csv (sum of `amount` for each resource, in MB).
    -   Incompatibility and prerequisite pairs: From export_07.csv and export_11.csv.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (total per-unit benefit in USD cents) × x[i]
    -   Minus: sum of item activation fees for each item selected (i.e., for each i with x[i] > 0, deduct item_fee)
    -   Minus: sum of category activation fees for each category used (i.e., for each c with any x[i] > 0 for i in c, deduct category activation_fee_cents)
    -   Plus: sum of bundle bonuses for each bundle where both items are selected (i.e., both x[item_a] > 0 and x[item_b] > 0)
7.  **Formulate Constraints:**
    -   **Authorization and Quantity Bounds:** For each item i:
        - If authorized = 0, x[i] = 0.
        - If authorized = 1, x[i] ∈ {0} ∪ [minimum_lot, maximum_order] (integer).
    -   **Item Activation Indicator:** For each item i:
        - z[i] = 1 if x[i] > 0, else 0. (Enforced via x[i] ≥ minimum_lot × z[i], x[i] ≤ maximum_order × z[i])
    -   **Category Quantity Bounds:** For each category c:
        - sum_{i in c} x[i] ≥ minimum_quantity[c] × y[c]
        - sum_{i in c} x[i] ≤ maximum_quantity[c] × y[c]
        - y[c] = 1 if any x[i] > 0 for i in c, else 0.
    -   **Platform Memory Capacity:** For each platform p:
        - sum_{i in p} (usage[i] in MB) × x[i] ≤ total available capacity for p (from capacity_ledger, after summing opening and reservation).
    -   **Incompatibility:** For each incompatible pair (i, j):
        - z[i] + z[j] ≤ 1 (cannot select both).
    -   **Prerequisite:** For each (i, prereq):
        - z[i] ≤ z[prereq] (if i is selected, prereq must also be selected).
    -   **Bundle Bonuses:** For each bundle (item_a, item_b):
        - b[bundle] ≤ z[item_a]
        - b[bundle] ≤ z[item_b]
        - b[bundle] ≥ z[item_a] + z[item_b] - 1 (b[bundle] = 1 iff both items selected)
    -   **Variable Domains:** All x[i] are integer, all z[i], y[c], b[bundle] are binary.
[Abstract Model Plan END]
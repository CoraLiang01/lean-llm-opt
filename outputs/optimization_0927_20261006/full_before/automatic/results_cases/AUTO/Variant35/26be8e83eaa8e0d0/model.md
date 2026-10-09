[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate licensed game-edition packages (item/platform options) to PC, CONSOLE, and MOBILE, maximizing net licensing return (benefit after fees and bonuses), subject to platform-specific memory capacity, per-category quantity bounds, item/category activation fees, incompatibility and prerequisite rules, bundle bonuses, and only allowing authorized options in integer lots within specified bounds.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, logical (incompatibility/prerequisite) constraints, and resource allocation.
3.  **Define Index Sets:** The primary indices are:
    - Items/options: all rows in export_08.csv (item_ref, each with platform/location_id and category)
    - Platforms: PC, CONSOLE, MOBILE (from location_id/resource columns)
    - Categories: all rows in export_04.csv (category)
    - Bundles: all rows in export_02.csv (pairs of item_refs)
    - Incompatibility pairs: all rows in export_07.csv (item_a, item_b)
    - Prerequisite pairs: all rows in export_11.csv (item_ref, prerequisite_ref)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item_ref i to allocate (must be 0 if unauthorized; otherwise, 0 or integer in [minimum_lot, maximum_order]). Type: GRB.INTEGER.
    -   `y[i]` = Binary variable: 1 if item_ref i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = Binary variable: 1 if any item in category c is selected (sum over i in c of y[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = Binary variable: 1 if both items in bundle are selected (for each bundle row), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients (benefit per unit): Calculated for each item_ref by summing all benefit components in export_01.csv for that item, converting each amount to USD cents using export_05.csv (amount * usd_cents_numerator / denominator), then summing.
    -   Item activation fees: export_09.csv (activation_fee_cents per item_ref).
    -   Category activation fees: export_04.csv (activation_fee_cents per category).
    -   Bundle bonuses: export_02.csv (bonus_cents per bundle).
    -   Memory usage per unit: export_12.csv (amount in GB per item_ref/resource, convert to MB by multiplying by 1000).
    -   Platform memory capacity: export_03.csv (sum of opening and reservation for each resource/platform, in MB).
    -   Category bounds: export_04.csv (minimum_quantity, maximum_quantity per category).
    -   Authorization, lot/order bounds, category, platform: export_08.csv (authorized, minimum_lot, maximum_order, category, location_id per item_ref).
    -   Incompatibility: export_07.csv (item_a, item_b pairs).
    -   Prerequisites: export_11.csv (item_ref, prerequisite_ref pairs).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (benefit per unit * x[i]) 
    -   Minus sum over all items: (item activation_fee_cents * y[i]) 
    -   Minus sum over all categories: (category activation_fee_cents * z[c])
    -   Plus sum over all bundles: (bonus_cents * b[bundle])
7.  **Formulate Constraints:**
    -   **Authorization and Lot/Order Bounds:** For each item_ref i:
        - If authorized = 0: x[i] = 0.
        - If authorized = 1: x[i] = 0 or x[i] in [minimum_lot, maximum_order] (integer).
        - y[i] = 1 if x[i] > 0, else 0.
    -   **Platform Memory Capacity:** For each platform p (PC, CONSOLE, MOBILE):
        - Sum over all items i assigned to p: (memory usage per unit[i] * x[i]) ≤ total available MB for p (from export_03.csv).
    -   **Category Quantity Bounds:** For each category c:
        - Sum over all items i in c: minimum_quantity[c] ≤ sum(x[i]) ≤ maximum_quantity[c].
        - z[c] = 1 if any y[i] in c is 1, else 0.
    -   **Item Activation Fee Linking:** For each item i: y[i] = 1 if x[i] > 0, else 0.
    -   **Category Activation Fee Linking:** For each category c: z[c] = 1 if any y[i] in c is 1, else 0.
    -   **Incompatibility:** For each incompatible pair (i, j): y[i] + y[j] ≤ 1.
    -   **Prerequisite:** For each (i, prereq): y[i] ≤ y[prereq].
    -   **Bundle Bonuses:** For each bundle (item_a, item_b): b[bundle] = 1 if y[item_a] = 1 and y[item_b] = 1, else 0.
    -   **Variable Domains:** x[i] integer, y[i], z[c], b[bundle] binary.
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate licensed game-edition packages (item_ref) to PC, CONSOLE, and MOBILE platforms, maximizing net licensing return (benefit after fees and bonuses), subject to platform-specific memory capacity, item/category quantity bounds, authorization, incompatibility, prerequisite, and bundle bonus rules. All data is to be taken directly from the supplied tables.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items: All item_ref from the item table (export_08.csv).
    - Platforms: PC, CONSOLE, MOBILE (from location_id/resource fields).
    - Categories: All category from the category table (export_04.csv).
    - Bundles: All (item_a, item_b) pairs from the bundle table (export_02.csv).
    - Incompatibility pairs: All (item_a, item_b) from the incompatible table (export_07.csv).
    - Prerequisite pairs: All (item_ref, prerequisite_ref) from the requires table (export_11.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item i to allocate (item_ref). Type: GRB.INTEGER. Domain: 0 or any integer in [minimum_lot[i], maximum_order[i]] if authorized; 0 if not authorized.
    -   `y[i]` = Binary variable: 1 if item i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = Binary variable: 1 if any item in category c is selected (i.e., category is active), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = Binary variable: 1 if both items in bundle (item_a, item_b) are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit per unit for each item: sum of all benefit table (export_01.csv) rows for item_ref, each amount converted to USD cents using fx table (export_05.csv): amount * usd_cents_numerator / denominator.
    -   Item activation fee: activation_fee_cents from item_fee table (export_09.csv), applied once per item if x[i] > 0.
    -   Category activation fee: activation_fee_cents from category table (export_04.csv), applied once per category if any item in the category is selected.
    -   Bundle bonus: bonus_cents from bundle table (export_02.csv), applied once per bundle if both items are selected.
    -   Memory usage per unit: amount (in GB) from usage table (export_12.csv) for each item_ref and resource (platform), converted to MB (1 GB = 1000 MB).
    -   Platform memory capacity: sum of amount from capacity_ledger table (export_03.csv) for each resource (platform), in MB.
    -   Item authorization, minimum_lot, maximum_order, category: from item table (export_08.csv).
    -   Category quantity bounds: minimum_quantity, maximum_quantity from category table (export_04.csv).
    -   Incompatibility and prerequisite relations: from incompatible (export_07.csv) and requires (export_11.csv) tables.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (benefit per unit[i] * x[i]) 
    -   Minus sum over all selected items: item activation_fee_cents[i] * y[i]
    -   Minus sum over all active categories: category activation_fee_cents[c] * z[c]
    -   Plus sum over all bundles: bonus_cents[bundle] * b[bundle]
7.  **Formulate Constraints:**
    -   **Item Authorization and Bounds:** For each item i:
        - If authorized[i] == 1: x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer).
        - If authorized[i] == 0: x[i] = 0.
        - y[i] = 1 if x[i] > 0, y[i] = 0 if x[i] = 0.
    -   **Category Quantity Bounds:** For each category c:
        - sum over items i in category c of x[i] ≥ minimum_quantity[c]
        - sum over items i in category c of x[i] ≤ maximum_quantity[c]
        - z[c] = 1 if any y[i] = 1 for i in c; z[c] = 0 otherwise.
    -   **Platform Memory Capacity:** For each platform p:
        - sum over items i assigned to platform p of (usage_per_unit[i,p] * x[i]) ≤ total_capacity[p] (all in MB).
    -   **Incompatibility:** For each incompatible pair (i, j):
        - y[i] + y[j] ≤ 1 (cannot select both).
    -   **Prerequisite:** For each (i, prereq) in requires:
        - y[i] ≤ y[prereq] (if i is selected, prereq must be selected; no proportionality).
    -   **Bundle Bonus:** For each bundle (i, j):
        - b[bundle] ≤ y[i]
        - b[bundle] ≤ y[j]
        - b[bundle] ≥ y[i] + y[j] - 1 (b[bundle] = 1 iff both y[i] = y[j] = 1).
    -   **Variable Domains:** All variables as defined above.
[Abstract Model Plan END]
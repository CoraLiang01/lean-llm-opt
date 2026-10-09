[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate licensed game-edition packages (item_ref) to PC, CONSOLE, and MOBILE platforms, maximizing net licensing return (benefit after fees and bonuses), subject to per-platform memory capacity, per-category quantity bounds, item/category activation fees, authorization, minimum/maximum order sizes, incompatibility and prerequisite (requires) constraints, and bundle bonuses. All quantities are integer, and all monetary values are in USD cents after currency conversion.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, requires, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items: All item_ref from the item table (export_08.csv), filtered to authorized=1.
    - Platforms: location_id (PC, CONSOLE, MOBILE), as given in the item table.
    - Categories: category from the category table (export_04.csv).
    - Bundles: (item_a, item_b) pairs from the bundle table (export_02.csv).
    - Incompatibilities: (item_a, item_b) pairs from the incompatible table (export_07.csv).
    - Requires: (item_ref, prerequisite_ref) pairs from the requires table (export_11.csv).
    - Resources: resource from the usage and capacity_ledger tables (PC, CONSOLE, MOBILE).
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity of item i (item_ref) to allocate (0 or in [minimum_lot, maximum_order] if selected; 0 if unauthorized). Type: GRB.INTEGER.
    -   `z[i]` = Binary variable: 1 if item i is selected (q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `g[c]` = Binary variable: 1 if any item in category c is selected (at least one z[i]=1 for i in c), 0 otherwise. Type: GRB.BINARY.
    -   `b[a,b]` = Binary variable: 1 if both items a and b in a bundle are selected (z[a]=z[b]=1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit per unit for each item: sum of all benefit table (export_01.csv) rows for item_ref i, each amount converted to USD cents using fx table (export_05.csv): amount * usd_cents_numerator / denominator.
    -   Item activation fee: activation_fee_cents from item_fee table (export_09.csv), per item_ref.
    -   Category bounds and activation fee: minimum_quantity, maximum_quantity, activation_fee_cents from category table (export_04.csv), per category.
    -   Bundle bonus: bonus_cents from bundle table (export_02.csv), per (item_a, item_b) pair.
    -   Resource usage per unit: amount (converted to MB if needed) from usage table (export_12.csv), per item_ref and resource.
    -   Resource capacity: sum of amount from capacity_ledger table (export_03.csv), per resource (already in MB).
    -   Authorization, minimum_lot, maximum_order, category, location_id: from item table (export_08.csv), per item_ref.
    -   Incompatibility pairs: from incompatible table (export_07.csv).
    -   Requires pairs: from requires table (export_11.csv).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (benefit per unit[i] * q[i]) 
    -   Minus sum over all selected items: item activation_fee_cents[i] * z[i]
    -   Minus sum over all used categories: category activation_fee_cents[c] * g[c]
    -   Plus sum over all selected bundles: bonus_cents[a,b] * b[a,b]
7.  **Formulate Constraints:**
    -   **Item selection and quantity bounds:** For each authorized item i: q[i] = 0 or q[i] in [minimum_lot[i], maximum_order[i]]; enforce q[i] >= minimum_lot[i] * z[i], q[i] <= maximum_order[i] * z[i], q[i] = 0 if unauthorized.
    -   **Resource (memory) capacity per platform:** For each resource r (PC, CONSOLE, MOBILE): sum over items i on r of (usage_per_unit[i,r] * q[i]) <= total_capacity[r] (all in MB).
    -   **Category quantity bounds:** For each category c: sum over items i in c of q[i] >= minimum_quantity[c], sum <= maximum_quantity[c].
    -   **Category activation:** For each category c: g[c] >= z[i] for all i in c; g[c] = 1 iff at least one item in c is selected.
    -   **Item activation:** For each item i: z[i] = 1 iff q[i] > 0; z[i] in {0,1}.
    -   **Bundle selection:** For each bundle (a,b): b[a,b] <= z[a], b[a,b] <= z[b], b[a,b] >= z[a] + z[b] - 1; b[a,b] = 1 iff both items are selected.
    -   **Incompatibility:** For each incompatible pair (a,b): z[a] + z[b] <= 1.
    -   **Requires:** For each (item_ref, prerequisite_ref): z[item_ref] <= z[prerequisite_ref] (if item_ref is selected, prerequisite_ref must also be selected).
    -   **Integrality:** All q[i] are integer, all z[i], g[c], b[a,b] are binary.
[Abstract Model Plan END]
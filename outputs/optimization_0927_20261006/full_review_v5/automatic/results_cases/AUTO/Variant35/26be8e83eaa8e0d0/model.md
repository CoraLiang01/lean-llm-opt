[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate licensed game-edition packages (item_ref) to PC, CONSOLE, and MOBILE platforms, maximizing net licensing return (benefit after fees and bonuses), subject to per-platform memory capacity, item and category quantity bounds, authorization, incompatibility, prerequisite, and bundle bonus rules. All data rows are to be used as given; only authorized options may be selected, and all units and currencies must be converted as specified.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (item_ref) from the item table (all rows).
    - Platforms (location_id/resource: PC, CONSOLE, MOBILE).
    - Categories (category).
    - Bundles (pairs of item_refs from the bundle table).
4.  **Define Decision Variables:**
    - `q[i]` = Integer quantity of item_ref i to allocate (0 or in [minimum_lot, maximum_order] if authorized; 0 if unauthorized). Type: GRB.INTEGER.
    - `z[i]` = Binary variable: 1 if item_ref i is selected (q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    - `g[c]` = Binary variable: 1 if any item in category c is selected (sum of z[i] for items in c ≥ 1), 0 otherwise. Type: GRB.BINARY.
    - `b[pair]` = Binary variable: 1 if both items in bundle pair are selected (z[i_a] = z[i_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Benefit per unit for each item: sum of all benefit table rows for item_ref i, each amount converted to USD cents using fx table (amount * usd_cents_numerator / denominator).
    - Item activation fee: item_fee table, activation_fee_cents per item_ref.
    - Bundle bonus: bundle table, bonus_cents per (item_a, item_b) pair.
    - Category bounds and activation fee: category table, minimum_quantity, maximum_quantity, activation_fee_cents per category.
    - Item bounds and eligibility: item table, authorized (1/0), minimum_lot, maximum_order, category, location_id.
    - Resource usage per item: usage table, amount (convert GB to MB by multiplying by 1000), per item_ref and resource.
    - Platform memory capacity: sum of capacity_ledger table rows per resource (PC, CONSOLE, MOBILE), in MB.
    - Incompatibility: incompatible table, pairs of item_refs that cannot both be selected.
    - Prerequisite: requires table, item_ref and prerequisite_ref pairs (if item_ref selected, prerequisite_ref must also be selected).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over all items: (benefit per unit in USD cents) × q[i]
    - Minus: sum over all items: item activation_fee_cents × z[i]
    - Minus: sum over all categories: category activation_fee_cents × g[c]
    - Plus: sum over all bundles: bonus_cents × b[pair]
7.  **Formulate Constraints:**
    - **Item selection and bounds:** For each item_ref i:
        - If authorized = 1: q[i] ∈ {0} ∪ [minimum_lot, maximum_order], integer; enforce q[i] ≥ minimum_lot × z[i], q[i] ≤ maximum_order × z[i], q[i] ≤ (maximum_order if authorized else 0).
        - If authorized = 0: q[i] = 0, z[i] = 0.
        - z[i] = 1 if q[i] > 0, 0 otherwise.
    - **Platform (resource) capacity:** For each platform/resource r (PC, CONSOLE, MOBILE):
        - Sum over all items assigned to r: (usage amount in MB per unit) × q[i] ≤ total available capacity for r (sum of capacity_ledger amounts for r).
    - **Category quantity bounds:** For each category c:
        - Sum over all items in c: q[i] ≥ minimum_quantity[c] × g[c]
        - Sum over all items in c: q[i] ≤ maximum_quantity[c] × g[c]
        - For all items i in c: z[i] ≤ g[c]; g[c] ≤ sum over i in c of z[i]
    - **Incompatibility:** For each incompatible pair (i, j): z[i] + z[j] ≤ 1
    - **Prerequisite:** For each (item_ref i, prerequisite_ref j): z[i] ≤ z[j]
    - **Bundle bonuses:** For each bundle pair (i, j): b[pair] ≤ z[i], b[pair] ≤ z[j], b[pair] ≥ z[i] + z[j] - 1
    - **Variable domains:** q[i] integer, z[i], g[c], b[pair] binary.
[Abstract Model Plan END]
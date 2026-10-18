[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate licensed game-edition packages (item_ref) to PC, CONSOLE, and MOBILE platforms, maximizing net licensing return (benefit after fees and bonuses), subject to per-platform memory capacity, item/category constraints, incompatibility and prerequisite rules, and bundle bonuses. Only authorized options may be chosen, and each item/category has specific quantity and fee rules.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (item_ref) from the item table (all rows).
    - Platforms (location_id/resource): PC, CONSOLE, MOBILE.
    - Categories (category) from the category table (all rows).
    - Bundles (pairs of item_refs) from the bundle table (all rows).
    - Incompatibility pairs (item_a, item_b) from the incompatible table (all rows).
    - Prerequisite pairs (item_ref, prerequisite_ref) from the requires table (all rows).
4.  **Define Decision Variables:**
    - `x[i]` = integer quantity of item_ref i to allocate (0 or in [minimum_lot, maximum_order] if authorized, else 0). Type: GRB.INTEGER.
    - `y[i]` = 1 if item_ref i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    - `z[c]` = 1 if any item in category c is selected (i.e., category is active), 0 otherwise. Type: GRB.BINARY.
    - `b[pair]` = 1 if both items in bundle pair are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Benefit per unit: For each item_ref, sum all benefit table rows for that item, convert each amount to USD cents using fx table (amount * usd_cents_numerator / denominator), then sum.
    - Item activation fee: activation_fee_cents from item_fee table, per item_ref.
    - Category activation fee: activation_fee_cents from category table, per category.
    - Bundle bonus: bonus_cents from bundle table, per bundle pair.
    - Memory usage per unit: amount (in GB) from usage table, per item_ref and resource; convert to MB (1 GB = 1000 MB).
    - Platform memory capacity: sum of capacity_ledger entries (opening + reservation) per resource (PC, CONSOLE, MOBILE), in MB.
    - Item eligibility and bounds: authorized, minimum_lot, maximum_order, category, location_id from item table.
    - Category quantity bounds: minimum_quantity, maximum_quantity from category table.
    - Incompatibility: item_a, item_b from incompatible table.
    - Prerequisites: item_ref, prerequisite_ref from requires table.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over all items: (benefit per unit * x[i]) 
    - Minus sum over all selected items: item activation_fee_cents * y[i]
    - Minus sum over all active categories: category activation_fee_cents * z[c]
    - Plus sum over all bundle pairs: bonus_cents * b[pair]
7.  **Formulate Constraints:**
    - **Item selection and bounds:** For each item_ref:
        - If authorized = 1: x[i] ∈ {0} ∪ [minimum_lot, maximum_order], integer.
        - If authorized = 0: x[i] = 0.
        - y[i] = 1 iff x[i] > 0; enforce x[i] ≥ minimum_lot * y[i], x[i] ≤ maximum_order * y[i], y[i] ∈ {0,1}.
    - **Platform memory capacity:** For each platform/resource:
        - Sum over items assigned to that platform: (usage per unit in MB) * x[i] ≤ total available MB for that platform (sum of capacity_ledger entries for that resource).
    - **Category quantity bounds and activation:** For each category c:
        - Sum over items in c: x[i] ≥ minimum_quantity[c] and ≤ maximum_quantity[c].
        - z[c] = 1 iff any y[i] for items in c is 1; enforce y[i] ≤ z[c] for all i in c, z[c] ≤ sum(y[i] for i in c).
    - **Incompatibility:** For each incompatible pair (i, j):
        - y[i] + y[j] ≤ 1.
    - **Prerequisites:** For each (i, prereq):
        - y[i] ≤ y[prereq].
    - **Bundle bonuses:** For each bundle pair (i, j):
        - b[pair] = 1 iff y[i] = 1 and y[j] = 1; enforce b[pair] ≤ y[i], b[pair] ≤ y[j], b[pair] ≥ y[i] + y[j] - 1.
    - **Non-negativity and integrality:** All x[i] ≥ 0, integer; all y[i], z[c], b[pair] ∈ {0,1}.
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate licensed game-edition packages (item/platform options) to PC, CONSOLE, and MOBILE, maximizing net licensing return (benefit after all fees and bonuses), subject to per-platform memory capacity, per-category quantity bounds, item/category activation fees, incompatibility and prerequisite rules, bundle bonuses, and only using authorized options. All data is to be taken directly from the supplied tables.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, and logical (incompatibility/prerequisite) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items/options (`item_ref` from the item table; each is a specific edition-platform combination)
    - Platforms (`location_id`/`resource`: PC, CONSOLE, MOBILE)
    - Categories (`category`)
    - Bundles (pairs of items from the bundle table)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item option `i` (item_ref) to allocate. Type: GRB.INTEGER. Domain: {0} or {minimum_lot[i], ..., maximum_order[i]} if authorized; 0 if unauthorized.
    -   `z[i]` = Binary variable: 1 if item option `i` is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[c]` = Binary variable: 1 if any item in category `c` is selected (i.e., sum of x[i] for items in c > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = Binary variable: 1 if both items in bundle are selected (i.e., both z[item_a] and z[item_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   **Benefit per unit:** For each item_ref, sum all benefit components (from benefit table) after converting each amount to USD cents using the fx table (amount * usd_cents_numerator / denominator). This is the per-unit benefit in USD cents.
    -   **Item activation fee:** From item_fee table, activation_fee_cents per item_ref (deducted once if x[i] > 0).
    -   **Category activation fee:** From category table, activation_fee_cents per category (deducted once if any item in category is selected).
    -   **Bundle bonus:** From bundle table, bonus_cents per bundle (added once if both items are selected).
    -   **Memory usage per unit:** From usage table, amount (in GB) per item_ref per resource; convert to MB (1 GB = 1000 MB).
    -   **Platform memory capacity:** For each resource (PC, CONSOLE, MOBILE), sum of capacity_ledger entries (opening + reservation) in MB.
    -   **Category quantity bounds:** From category table, minimum_quantity and maximum_quantity per category (sum of x[i] for items in category).
    -   **Authorization, lot/order bounds:** From item table, authorized (1/0), minimum_lot, maximum_order per item_ref.
    -   **Incompatibility:** From incompatible table, pairs of item_refs that cannot both be selected.
    -   **Prerequisites:** From requires table, item_ref requires prerequisite_ref (if x[item_ref] > 0, then x[prerequisite_ref] > 0).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (per-unit benefit[i] * x[i])
    -   Minus: sum over all items: item activation_fee_cents[i] * z[i]
    -   Minus: sum over all categories: category activation_fee_cents[c] * w[c]
    -   Plus: sum over all bundles: bonus_cents[bundle] * b[bundle]
7.  **Formulate Constraints:**
    -   **Authorization and bounds:** For each item_ref:
        - If authorized = 0: x[i] = 0
        - If authorized = 1: x[i] = 0 or x[i] in [minimum_lot[i], maximum_order[i]] (i.e., x[i] = 0 or x[i] ≥ minimum_lot[i] and x[i] ≤ maximum_order[i])
        - z[i] = 1 if x[i] > 0, 0 otherwise (enforced via x[i] ≤ maximum_order[i] * z[i], x[i] ≥ minimum_lot[i] * z[i] for authorized items)
    -   **Platform memory capacity:** For each platform/resource r (PC, CONSOLE, MOBILE):
        - sum over items assigned to r: (usage[i,r] in MB) * x[i] ≤ total available capacity for r (sum of capacity_ledger entries for r)
    -   **Category quantity bounds:** For each category c:
        - sum over items in c: x[i] ≥ minimum_quantity[c] * w[c]
        - sum over items in c: x[i] ≤ maximum_quantity[c] * w[c]
        - w[c] = 1 if any x[i] > 0 for items in c, 0 otherwise
    -   **Incompatibility:** For each incompatible pair (i, j):
        - z[i] + z[j] ≤ 1
    -   **Prerequisites:** For each (item_ref, prerequisite_ref):
        - z[item_ref] ≤ z[prerequisite_ref] (i.e., if item_ref is selected, prerequisite_ref must also be selected)
    -   **Bundle bonuses:** For each bundle (item_a, item_b):
        - b[bundle] ≤ z[item_a]
        - b[bundle] ≤ z[item_b]
        - b[bundle] ≥ z[item_a] + z[item_b] - 1 (i.e., b[bundle] = 1 iff both z[item_a] and z[item_b] = 1)
    -   **Variable domains:** x[i] integer, z[i], w[c], b[bundle] binary.
[Abstract Model Plan END]
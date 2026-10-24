[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate licensed game-edition packages (item/platform options) to PC, CONSOLE, and MOBILE, maximizing net licensing return (benefit after fees and bonuses), subject to per-platform memory capacity, per-category quantity bounds, item/category activation fees, item authorization, minimum/maximum order sizes, incompatibility and prerequisite (requires) constraints, and bundle bonuses. All data is to be taken directly from the supplied tables, with all units and currency conversions as specified.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, requires, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items/options: all rows in the item table (item_ref).
    - Platforms/resources: PC, CONSOLE, MOBILE (from location_id/resource fields).
    - Categories: all rows in the category table (category).
    - Bundles: all rows in the bundle table (item_a, item_b).
    - Incompatibility pairs: all rows in the incompatible table (item_a, item_b).
    - Requires pairs: all rows in the requires table (item_ref, prerequisite_ref).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item option `i` (item_ref) to allocate. Type: GRB.INTEGER. Domain: 0 or any integer in [minimum_lot, maximum_order] if authorized; 0 if unauthorized.
    -   `y[i]` = Binary variable: 1 if item option `i` is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = Binary variable: 1 if any item in category `c` is selected (sum of y[i] for items in c ≥ 1), 0 otherwise. Type: GRB.BINARY.
    -   `b[p]` = Binary variable: 1 if both items in bundle pair `p` are selected (y[item_a] = y[item_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit per unit: For each item_ref, sum all benefit table rows for that item, convert each amount to USD cents using fx table (amount * usd_cents_numerator / denominator), and sum.
    -   Item activation fee: From item_fee table, activation_fee_cents per item_ref.
    -   Category bounds and activation fee: From category table, minimum_quantity, maximum_quantity, activation_fee_cents per category.
    -   Bundle bonus: From bundle table, bonus_cents per (item_a, item_b) pair.
    -   Memory usage per unit: From usage table, amount (convert GB to MB as needed) per item_ref/resource.
    -   Platform memory capacity: For each resource, sum capacity_ledger table rows (opening + reservation) for that resource (in MB).
    -   Authorization, minimum_lot, maximum_order, category, location_id: From item table, per item_ref.
    -   Incompatibility: From incompatible table, pairs of item_refs.
    -   Requires: From requires table, item_ref and prerequisite_ref pairs.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (benefit per unit * x[i]) 
    -   Minus sum over all items: (item activation_fee_cents * y[i]) 
    -   Minus sum over all categories: (category activation_fee_cents * z[c])
    -   Plus sum over all bundles: (bonus_cents * b[p])
7.  **Formulate Constraints:**
    -   **Item selection and bounds:** For each item_ref:
        - If authorized = 1: x[i] ∈ {0} ∪ [minimum_lot, maximum_order]; enforce x[i] ≥ minimum_lot * y[i], x[i] ≤ maximum_order * y[i], y[i] ∈ {0,1}.
        - If authorized = 0: x[i] = 0, y[i] = 0.
    -   **Platform memory capacity:** For each resource/platform:
        - Sum over all items assigned to that platform: (usage per unit in MB * x[i]) ≤ total available MB for that platform (from capacity_ledger).
    -   **Category quantity bounds:** For each category:
        - Sum over all items in the category: minimum_quantity ≤ sum(x[i]) ≤ maximum_quantity.
        - z[c] = 1 if any y[i] for items in c is 1; z[c] = 0 otherwise.
    -   **Item activation fee logic:** For each item_ref: y[i] = 1 iff x[i] > 0.
    -   **Category activation fee logic:** For each category: z[c] = 1 iff any y[i] in c is 1.
    -   **Incompatibility:** For each incompatible pair (item_a, item_b): y[item_a] + y[item_b] ≤ 1.
    -   **Requires:** For each requires pair (item_ref, prerequisite_ref): y[item_ref] ≤ y[prerequisite_ref].
    -   **Bundle bonus logic:** For each bundle (item_a, item_b): b[p] ≤ y[item_a], b[p] ≤ y[item_b], b[p] ≥ y[item_a] + y[item_b] - 1.
[Abstract Model Plan END]
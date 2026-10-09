[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal number of packs to display for each product-and-section option (item_ref, location_id) in MARKET_SQUARE, maximizing net merchandising benefit. The plan must respect section capacities, per-option authorization and lot/order limits, resource usage, category quantity bounds, fixed activation fees (per option and per category), incompatibility and prerequisite (requires) relationships, and bundle bonuses. All data is to be taken directly from the supplied tables; only listed options are considered.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, requires, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Options: Each item_ref-location_id pair (from the union of the two item tables; each row is an option).
    - Categories: Each unique category (from the category table).
    - Resources: Each unique resource (from the capacity_ledger and usage tables).
    - Bundles: Each (item_a, item_b) pair from the bundle table.
    - Incompatibility pairs: Each (item_a, item_b) from the incompatible table.
    - Requires pairs: Each (item_ref, prerequisite_ref) from the requires table.
4.  **Define Decision Variables:**
    -   `x[o]` = Number of packs to display for option o (item_ref-location_id). Type: GRB.INTEGER, domain: {0} or [minimum_lot, maximum_order] if authorized.
    -   `y[o]` = 1 if option o is selected (x[o] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any option in category c is selected (category is "used"), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both options in bundle are selected (for bundle bonus), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each option: sum of amount_cents for each item_ref from the benefit table.
    -   Option activation fee: activation_fee_cents from item_fee table, per item_ref.
    -   Category activation fee: activation_fee_cents from category table, per category.
    -   Bundle bonus: bonus_cents from bundle table, per (item_a, item_b) pair.
    -   Option authorization, minimum_lot, maximum_order, category, location_id: from item tables (batch_01/export_07.csv and batch_02/export_08.csv).
    -   Resource usage per option: amount (converted to ml if needed) from usage tables, per (item_ref, resource).
    -   Section (resource) capacity: sum of amount (ml) from capacity_ledger table, per resource.
    -   Category quantity bounds: minimum_quantity, maximum_quantity from category table, per category.
    -   Incompatibility: pairs from incompatible table.
    -   Requires: pairs from requires table.
6.  **Formulate Objective:** Maximize total net merchandising benefit, defined as:
    -   Sum over all options of (per-unit benefit * x[o])
    -   Minus sum over all options of (option activation fee * y[o]) [fee charged once per option if used]
    -   Minus sum over all categories of (category activation fee * z[c]) [fee charged once per category if used]
    -   Plus sum over all bundles of (bundle bonus * b[bundle]) [bonus counted once if both options in bundle are selected]
    -   All amounts in USD cents.
7.  **Formulate Constraints:**
    -   **Option Authorization and Lot/Order Limits:** For each option o:
        - If authorized = 0, x[o] = 0.
        - If authorized = 1, x[o] ∈ {0} ∪ [minimum_lot, maximum_order] (integer).
        - y[o] = 1 if x[o] > 0, y[o] = 0 if x[o] = 0. (Enforced via x[o] ≥ minimum_lot * y[o], x[o] ≤ maximum_order * y[o])
    -   **Resource (Section) Capacity:** For each resource r (section):
        - sum over all options o assigned to r of (resource usage per pack * x[o]) ≤ total available capacity for r (sum of capacity_ledger entries for r, after converting all units to ml).
    -   **Category Quantity Bounds:** For each category c:
        - sum over all options o in category c of x[o] ≥ minimum_quantity[c] * z[c]
        - sum over all options o in category c of x[o] ≤ maximum_quantity[c] * z[c]
        - z[c] = 1 if any x[o] > 0 for o in c, 0 otherwise.
    -   **Incompatibility:** For each incompatible pair (item_a, item_b):
        - y[item_a] + y[item_b] ≤ 1 (cannot select both).
    -   **Requires:** For each requires pair (item_ref, prerequisite_ref):
        - y[item_ref] ≤ y[prerequisite_ref] (if item_ref is selected, prerequisite_ref must also be selected; no proportionality).
    -   **Bundle Bonuses:** For each bundle (item_a, item_b):
        - b[bundle] ≤ y[item_a]
        - b[bundle] ≤ y[item_b]
        - b[bundle] ≥ y[item_a] + y[item_b] - 1 (b[bundle] = 1 iff both y[item_a] = y[item_b] = 1)
    -   **Variable Domains:** All x[o] are integer, y[o], z[c], b[bundle] are binary.
[Abstract Model Plan END]
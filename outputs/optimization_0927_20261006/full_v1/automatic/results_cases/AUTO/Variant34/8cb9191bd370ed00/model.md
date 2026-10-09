[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer number of packs to display for each authorized product-and-section option (item_ref), maximizing total net merchandising benefit (in USD cents), subject to section capacities, per-option and per-category limits, incompatibility and prerequisite rules, and bundle bonuses. The plan must deduct fixed item and category activation fees as specified, and only authorized options may be selected.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, requires, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Options: set of item_ref (product-and-section options) from the union of all item tables (all rows used).
    - Sections: set of location_id (display sections) from item tables.
    - Categories: set of category from category table.
    - Resources: set of resource from capacity_ledger and usage tables.
    - Bundles: set of (item_a, item_b) pairs from bundle table.
    - Incompatibles: set of (item_a, item_b) pairs from incompatible table.
    - Requires: set of (item_ref, prerequisite_ref) pairs from requires table.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of packs of option i (item_ref) to display. Type: GRB.INTEGER, domain: {0} or [minimum_lot, maximum_order] if authorized, 0 if unauthorized.
    -   `y[i]` = 1 if option i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any option in category c is selected (sum over i in c of y[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both options in bundle are selected (y[item_a] = y[item_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each option: sum of amount_cents for each item_ref in benefit table.
    -   Item activation fee: activation_fee_cents from item_fee table, applied once per option if selected.
    -   Category activation fee: activation_fee_cents from category table, applied once per category if any option in that category is selected.
    -   Section capacity: sum of amount (converted to ml if needed) for each resource (section) from capacity_ledger table.
    -   Per-pack resource usage: amount (converted to ml if needed) for each item_ref and resource from usage tables.
    -   Option bounds: minimum_lot and maximum_order from item tables, only if authorized = 1; otherwise, x[i] = 0.
    -   Category quantity limits: minimum_quantity and maximum_quantity from category table, sum over all options in category.
    -   Incompatibility pairs: from incompatible table.
    -   Requires pairs: from requires table.
    -   Bundle bonuses: bonus_cents from bundle table.
6.  **Formulate Objective:** Maximize total net merchandising benefit, defined as:
    -   Sum over all options of (per-unit benefit * x[i])
    -   Minus sum over all selected options of item activation_fee_cents * y[i]
    -   Minus sum over all activated categories of category activation_fee_cents * z[c]
    -   Plus sum over all activated bundles of bonus_cents * b[bundle]
    -   All terms in USD cents.
7.  **Formulate Constraints:**
    -   Option authorization and bounds: For each option i, x[i] = 0 if unauthorized; if authorized, x[i] = 0 or x[i] in [minimum_lot, maximum_order] (integer).
    -   Section (resource) capacity: For each resource r, sum over all options i of (per-pack usage of r by i * x[i]) ≤ total available capacity of r (sum of amount in capacity_ledger for r, converting units as needed).
    -   Category quantity limits: For each category c, sum over all options i in c of x[i] ≥ minimum_quantity[c] and ≤ maximum_quantity[c].
    -   Linking variables: For each option i, y[i] = 1 if x[i] > 0, 0 otherwise; for each category c, z[c] = 1 if any y[i] in c is 1, 0 otherwise.
    -   Incompatibility: For each (item_a, item_b) in incompatible, y[item_a] + y[item_b] ≤ 1.
    -   Requires: For each (item_ref, prerequisite_ref) in requires, y[item_ref] ≤ y[prerequisite_ref].
    -   Bundle activation: For each bundle (item_a, item_b), b[bundle] = 1 if y[item_a] = y[item_b] = 1, 0 otherwise.
[Abstract Model Plan END]
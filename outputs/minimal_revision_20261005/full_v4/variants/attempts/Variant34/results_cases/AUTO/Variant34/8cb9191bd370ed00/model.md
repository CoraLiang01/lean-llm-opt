[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal number of packs to display for each product-and-section option (item_ref at location_id) in MARKET_SQUARE, maximizing net merchandising benefit. The plan must respect section capacities, per-option and per-category constraints, authorization, incompatibility and prerequisite rules, and bundle bonuses, while accounting for fixed fees and resource usage.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, logical (incompatibility/prerequisite) constraints, and combinatorial bundle bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Item options: All item_ref-location_id pairs (from the union of both item tables; each row is an option).
    - Categories: All unique category values (from category table).
    - Resources: All unique resource values (from capacity_ledger and usage tables).
    - Bundles: All (item_a, item_b) pairs from the bundle table.
    - Incompatibility pairs: All (item_a, item_b) from the incompatible table.
    - Prerequisite pairs: All (item_ref, prerequisite_ref) from the requires table.
4.  **Define Decision Variables:**
    -   `x[o]` = Number of packs to display for option o (item_ref at location_id). Type: GRB.INTEGER, with bounds [0, maximum_order[o]] and minimum_lot[o] if x[o] > 0.
    -   `y[o]` = 1 if option o is selected (x[o] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any option in category c is selected (i.e., sum of x[o] for options in c > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both options in bundle are selected (i.e., both y[o_a] = 1 and y[o_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each option: sum of amount_cents for each item_ref in benefit table.
    -   Fixed activation fee per option: activation_fee_cents from item_fee table.
    -   Bundle bonus: bonus_cents from bundle table.
    -   Section capacity: sum of amount (converted to ml) for each resource in capacity_ledger table (opening + reservation).
    -   Per-pack resource usage: amount (converted to ml) from usage tables, by item_ref and resource.
    -   Option bounds: minimum_lot, maximum_order, and authorized from item tables.
    -   Category bounds and activation fee: minimum_quantity, maximum_quantity, activation_fee_cents from category table.
    -   Option-category mapping: from item tables.
    -   Option-location mapping: from item tables.
    -   Incompatibility and prerequisite pairs: from incompatible and requires tables.
6.  **Formulate Objective:** Maximize total net merchandising benefit, defined as:
    -   Sum over all options: (per-unit benefit[o] * x[o]) 
    -   Minus sum over all options: (item_fee[o] * y[o]) [fixed fee if option used]
    -   Minus sum over all categories: (activation_fee_cents[c] * z[c]) [category fee if any option in c used]
    -   Plus sum over all bundles: (bonus_cents[bundle] * b[bundle]) [bonus if both options in bundle are used]
    -   All terms in USD cents.
7.  **Formulate Constraints:**
    -   **Authorization:** For each option o, if authorized[o] == 0, x[o] = 0.
    -   **Option bounds:** For each option o, x[o] = 0 or minimum_lot[o] ≤ x[o] ≤ maximum_order[o]. Enforce x[o] = 0 if y[o] = 0, and x[o] ≥ minimum_lot[o] * y[o].
    -   **Section (resource) capacity:** For each resource r (section), sum over all options o assigned to r: (usage[o, r] * x[o]) ≤ total available capacity[r] (in ml).
    -   **Category quantity bounds:** For each category c, sum over all options o in c: minimum_quantity[c] ≤ sum(x[o]) ≤ maximum_quantity[c].
    -   **Category activation:** For each category c, z[c] = 1 if any x[o] > 0 for o in c, else 0.
    -   **Option activation:** For each option o, y[o] = 1 if x[o] > 0, else 0.
    -   **Incompatibility:** For each incompatible pair (o1, o2), y[o1] + y[o2] ≤ 1.
    -   **Prerequisite:** For each (o, prereq), y[o] ≤ y[prereq].
    -   **Bundle bonus:** For each bundle (o1, o2), b[bundle] = 1 if y[o1] = 1 and y[o2] = 1, else 0.
    -   **Variable domains:** x[o] ∈ {0} ∪ [minimum_lot[o], maximum_order[o]] (integers); y[o], z[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]
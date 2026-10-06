[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal number of packs to display for each product-and-section option (item_ref at location_id) in MARKET_SQUARE, maximizing net merchandising benefit. The plan must respect section capacities, item/category constraints, authorization, incompatibilities, prerequisites, and bundle bonuses, while accounting for fixed fees and resource usage.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, logical (incompatibility/prerequisite) constraints, and combinatorial bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Item options: All item_ref-location_id pairs (from the union of both item tables; each row is an option).
    - Categories: All unique category values from the item tables and category table.
    - Resources: All unique resource values from the usage and capacity_ledger tables (e.g., SECTION_A, SECTION_B, SECTION_C).
    - Bundles: All (item_a, item_b) pairs from the bundle table.
    - Incompatibilities: All (item_a, item_b) pairs from the incompatible table.
    - Prerequisites: All (item_ref, prerequisite_ref) pairs from the requires table.
4.  **Define Decision Variables:**
    -   `x[o]` = Number of packs to display for option o (item_ref at location_id). Type: GRB.INTEGER, with bounds [0, maximum_order] and minimum_lot if positive.
    -   `y[o]` = 1 if option o is selected (x[o] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any option in category c is selected (i.e., category is "active"), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both options in bundle are selected (for each bundle pair), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit: For each item_ref, sum all amount_cents from the benefit table (group by item_ref).
    -   Item activation fee: For each item_ref, from item_fee table (activation_fee_cents).
    -   Category activation fee: For each category, from category table (activation_fee_cents).
    -   Bundle bonus: For each bundle (item_a, item_b), from bundle table (bonus_cents).
    -   Resource usage: For each option, from usage tables (amount * 1000 if unit is 'liter', else as is), mapped to resource and item_ref.
    -   Section capacity: For each resource, sum all amount values from capacity_ledger table (convert all to ml).
    -   Option bounds: For each option, minimum_lot and maximum_order from item tables; authorized flag.
    -   Category bounds: For each category, minimum_quantity and maximum_quantity from category table.
    -   Incompatibilities: From incompatible table (pairs of item_refs).
    -   Prerequisites: From requires table (item_ref, prerequisite_ref).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all options: (per-unit benefit * x[o]) 
    -   Minus: sum of item activation_fee_cents for each option with x[o] > 0 (i.e., y[o] = 1)
    -   Minus: sum of category activation_fee_cents for each category with any x[o] > 0 (i.e., z[c] = 1)
    -   Plus: sum of bundle bonus_cents for each bundle where both options are selected (b[bundle] = 1)
7.  **Formulate Constraints:**
    -   **Authorization:** For each option, if authorized == 0, x[o] = 0.
    -   **Option bounds:** For each option, x[o] = 0 or x[o] in [minimum_lot, maximum_order] (if authorized).
    -   **Section (resource) capacity:** For each resource (section), sum over all options using that resource: (usage per pack * x[o]) ≤ total available capacity (sum of capacity_ledger entries for that resource, all in ml).
    -   **Category quantity bounds:** For each category, sum of x[o] over all options in that category ∈ [minimum_quantity, maximum_quantity].
    -   **Item activation linking:** For each option, y[o] = 1 if x[o] > 0, else 0. Enforced via: x[o] ≤ maximum_order * y[o]; x[o] ≥ minimum_lot * y[o] (if minimum_lot > 0).
    -   **Category activation linking:** For each category, z[c] = 1 if any x[o] > 0 for options in c, else 0. Enforced via: for all o in c, x[o] ≤ M * z[c]; sum over o in c of x[o] ≥ z[c] (if at least one must be positive for z[c]=1).
    -   **Incompatibility:** For each incompatible pair (item_a, item_b), y[a] + y[b] ≤ 1.
    -   **Prerequisite:** For each (item_ref, prerequisite_ref), y[item_ref] ≤ y[prerequisite_ref].
    -   **Bundle bonus linking:** For each bundle (item_a, item_b), b[bundle] ≤ y[item_a], b[bundle] ≤ y[item_b], b[bundle] ≥ y[item_a] + y[item_b] - 1.
    -   **Variable domains:** x[o] ∈ {0} ∪ [minimum_lot, maximum_order] (integer), y[o], z[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]
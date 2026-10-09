[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a portfolio of indivisible, authorized development modules (items) in New York to maximize net return (benefit minus setup charges), subject to resource (storage, labor, energy) limits, category quantity bounds, incompatibility and prerequisite logic, and bundle bonuses. Each item can be chosen in integer multiples within its allowed lot/order range, or not at all. Fixed fees apply per item and per category if used. Bundles provide bonuses if both items are selected. Unauthorized items must not be chosen.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge and logical (combinatorial) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (i ∈ I): All item_refs from the union of item tables (authorized subset only).
    - Categories (g ∈ G): All categories from the category table.
    - Resources (r ∈ R): All resources from the usage and capacity_ledger tables.
    - Bundles (b ∈ B): All bundle pairs from the bundle table.
    - Incompatibility pairs (p ∈ P): All item pairs from the incompatible table.
    - Requires pairs (q ∈ Q): All (item, prerequisite) pairs from the requires table.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item i selected (0 if not selected). Type: GRB.INTEGER, with bounds [minimum_lot[i], maximum_order[i]] if selected, 0 otherwise.
    -   `y[i]` = 1 if item i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category g is selected (category is used), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle b are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-item benefit: sum of amount_cents for each item_ref in the benefit table (all components for that item).
    -   Per-item fixed fee: activation_fee_cents from the item_fee table.
    -   Per-category bounds and fee: minimum_quantity, maximum_quantity, activation_fee_cents from the category table.
    -   Per-item resource usage: amount from usage table for each (item_ref, resource).
    -   Resource capacities: sum of amount for each resource in capacity_ledger table (opening + reservation).
    -   Bundle bonuses: bonus_cents from bundle table for each (item_a, item_b).
    -   Authorization, lot/order bounds, and category mapping: from item tables (authorized, minimum_lot, maximum_order, category).
    -   Incompatibility and requires logic: from incompatible and requires tables.
6.  **Formulate Objective:** Maximize total net benefit in cents:
    -   Sum over items: (per-item benefit × x[i]) 
    -   Minus: sum of item_fee for each item with y[i]=1
    -   Minus: sum of category activation_fee for each category with z[g]=1
    -   Plus: sum of bundle bonus_cents for each bundle where both items are selected (w[b]=1)
7.  **Formulate Constraints:**
    -   Resource limits: For each resource r, sum over items of (usage per unit × x[i]) ≤ total available from capacity_ledger for r.
    -   Item selection: For each item i, x[i] = 0 if not authorized; otherwise, x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]].
    -   Linking: For each item i, y[i] = 1 if x[i] > 0, 0 otherwise; enforce x[i] ≥ minimum_lot[i] × y[i] and x[i] ≤ maximum_order[i] × y[i].
    -   Category bounds: For each category g, sum of x[i] over items in g ≥ minimum_quantity[g] and ≤ maximum_quantity[g].
    -   Category activation: For each category g, z[g] = 1 if any x[i] > 0 for i in g, 0 otherwise.
    -   Incompatibility: For each incompatible pair (i, j), y[i] + y[j] ≤ 1.
    -   Requires: For each (i, prerequisite j), y[i] ≤ y[j].
    -   Bundle bonuses: For each bundle (i, j), w[b] ≤ y[i], w[b] ≤ y[j], w[b] ≥ y[i] + y[j] - 1.
[Abstract Model Plan END]
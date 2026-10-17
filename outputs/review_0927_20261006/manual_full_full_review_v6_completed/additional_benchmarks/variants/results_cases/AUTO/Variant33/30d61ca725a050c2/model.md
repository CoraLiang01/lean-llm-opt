[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a portfolio of authorized, indivisible development modules (items), maximizing total net benefit (sum of per-unit benefits, minus item and category activation fees, plus bundle bonuses), subject to resource (storage, labor, energy) limits, category quantity bounds, and logical constraints (incompatibilities, prerequisites, bundles). Each item can be chosen in integer multiples within its allowed lot/order bounds or not at all; unauthorized items must not be selected.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (compatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (i): All item_ref values from the union of item tables (authorized subset only).
    - Categories (g): All category values from the category table.
    - Resources (r): All resource values from the capacity_ledger and usage tables.
    - Bundles (b): All bundle rows (pairs of items) from the bundle table.
    - Incompatibility pairs (p): All incompatible item pairs from the incompatible table.
    - Prerequisite pairs (q): All requires rows (item, prerequisite) from the requires table.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item i selected (0 or in [minimum_lot, maximum_order] if authorized; 0 if unauthorized). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category g is selected (category is "active"), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle b are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit item benefit: sum of amount_cents for each item_ref from all benefit table rows.
    -   Item activation fee: activation_fee_cents from item_fee table, per item_ref.
    -   Bundle bonus: bonus_cents from bundle table, per (item_a, item_b) pair.
    -   Resource usage per item: amount from usage table, per (item_ref, resource).
    -   Resource capacity: sum of amount for each resource from capacity_ledger table (sum all entries for each resource).
    -   Category bounds and activation fee: minimum_quantity, maximum_quantity, activation_fee_cents from category table, per category.
    -   Item eligibility and bounds: authorized, minimum_lot, maximum_order from item tables, per item_ref.
    -   Incompatibility: item_a, item_b pairs from incompatible table.
    -   Prerequisites: item_ref, prerequisite_ref pairs from requires table.
6.  **Formulate Objective:** Maximize total net benefit in cents:
        - Sum over items: (per-unit benefit * x[i]) 
        - Minus sum over items: (item activation_fee_cents * y[i])
        - Minus sum over categories: (category activation_fee_cents * z[g])
        - Plus sum over bundles: (bonus_cents * w[b])
7.  **Formulate Constraints:**
    -   **Item selection and bounds:** For each item i:
        - If authorized: x[i] = 0 or x[i] in [minimum_lot, maximum_order]; enforce x[i] >= minimum_lot * y[i], x[i] <= maximum_order * y[i], y[i] in {0,1}.
        - If unauthorized: x[i] = 0, y[i] = 0.
    -   **Resource limits:** For each resource r:
        - Sum over items: (resource usage per unit for i, r) * x[i] ≤ total capacity for r (sum of capacity_ledger amounts for r).
    -   **Category quantity bounds:** For each category g:
        - Sum over items in g: x[i] ≥ minimum_quantity for g.
        - Sum over items in g: x[i] ≤ maximum_quantity for g.
        - For all items i in g: y[i] ≤ z[g]; z[g] ≤ sum over i in g of y[i].
    -   **Incompatibility:** For each incompatible pair (i, j): y[i] + y[j] ≤ 1.
    -   **Prerequisites:** For each (i, prerequisite j): y[i] ≤ y[j].
    -   **Bundle bonuses:** For each bundle (i, j): w[b] ≤ y[i], w[b] ≤ y[j], w[b] ≥ y[i] + y[j] - 1.
[Abstract Model Plan END]
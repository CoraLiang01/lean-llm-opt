[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a portfolio of standardized development modules (indivisible options) in New York to maximize net return (benefit minus setup charges), subject to resource (storage, labor, energy) limits, category quantity bounds, item/category activation fees, incompatibility and prerequisite (requires) logic, and bundle bonuses. Only authorized options may be chosen, and each item must be selected in integer multiples within its allowed lot/order range.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, logical (incompatibility and requires) constraints, and bundle bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items (item_ref) — the module options (from item tables, filtered to authorized=1).
    - Categories (category) — groups of items (from category tables).
    - Resources (resource) — e.g., space, labor, power (from usage/capacity_ledger).
    - Bundles (pairs of items eligible for bonus, from bundle table).
    - Incompatibility pairs (from incompatible table).
    - Requires pairs (from requires table).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item i selected (0 if not selected, else between minimum_lot and maximum_order for authorized items). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item i is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is selected (i.e., category is "active"), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected (for bundle bonus), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-item benefit: sum of all amount_cents for each item_ref in the benefit table (export_01.csv).
    -   Per-item activation fee: activation_fee_cents from item_fee table (export_09.csv).
    -   Per-category activation fee: activation_fee_cents from category table (export_04.csv).
    -   Per-item resource usage: amount for each (item_ref, resource) in usage tables (export_12.csv, export_13.csv).
    -   Resource capacities: sum of amount for each resource in capacity_ledger table (export_03.csv).
    -   Item authorization, minimum_lot, maximum_order, and category assignment: from item tables (export_07.csv, export_08.csv), filtered to authorized=1.
    -   Category quantity bounds: minimum_quantity, maximum_quantity from category table (export_04.csv).
    -   Incompatibility pairs: from incompatible table (export_06.csv).
    -   Requires pairs: from requires table (export_11.csv).
    -   Bundle bonuses: bonus_cents for each bundle (export_02.csv).
6.  **Formulate Objective:** Maximize total net benefit in cents:
    -   Sum over all items: (per-item benefit * x[i]) 
    -   Minus sum over all selected items: (item activation_fee_cents * y[i])
    -   Minus sum over all active categories: (category activation_fee_cents * z[c])
    -   Plus sum over all eligible bundles: (bonus_cents * b[bundle])
7.  **Formulate Constraints:**
    -   **Item selection and bounds:** For each authorized item i, x[i] = 0 or x[i] in [minimum_lot, maximum_order]; for unauthorized items, x[i] = 0.
    -   **Item activation indicator:** For each item i, y[i] = 1 if x[i] > 0, else 0; enforce with x[i] <= maximum_order[i] * y[i] and x[i] >= minimum_lot[i] * y[i] (for authorized items).
    -   **Resource limits:** For each resource r, sum over all items of (usage amount of r per unit of i * x[i]) ≤ total available capacity for r (from capacity_ledger).
    -   **Category quantity bounds:** For each category c, sum of x[i] over all items in c ≥ minimum_quantity[c] and ≤ maximum_quantity[c].
    -   **Category activation indicator:** For each category c, z[c] = 1 if any x[i] > 0 for i in c, else 0; enforce with x[i] ≤ maximum_order[i] * z[c] for all i in c.
    -   **Incompatibility:** For each incompatible pair (i, j), y[i] + y[j] ≤ 1.
    -   **Requires:** For each requires pair (i, prereq), y[i] ≤ y[prereq].
    -   **Bundle bonuses:** For each bundle (i, j), b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1; b[bundle] = 1 only if both items are selected and authorized.
    -   **Variable domains:** x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integers); y[i], z[c], b[bundle] ∈ {0,1}.
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a portfolio of standardized development modules (indivisible options) in New York to maximize net return (benefit minus setup charges), subject to resource (storage, labor, energy) limits, category quantity bounds, item/category activation fees, incompatibility and prerequisite logic, and bundle bonuses. Only authorized options may be chosen, and each item must be selected in integer multiples within its allowed lot/order range.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, generalized assignment, and logical constraints (incompatibility, prerequisites, bundles).
3.  **Define Index Sets:** The primary indices are:
    - Items (module options): all item_refs from the union of item tables (authorized subset only).
    - Categories: all categories from the category table.
    - Resources: all resources from the capacity_ledger and usage tables.
    - Bundles: all bundle rows (pairs of items with associated bonuses).
    - Incompatibility pairs: all incompatible item pairs.
    - Prerequisite pairs: all requires rows (item, prerequisite).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item i selected (0 if not selected). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item i is selected in any positive quantity, 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is selected (category is "used"), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected (for bundle bonus), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-item benefit: sum of amount_cents for each item_ref in the benefit table (all components for that item).
    -   Per-item activation fee: activation_fee_cents from item_fee table, by item_ref.
    -   Per-category bounds and activation fee: minimum_quantity, maximum_quantity, activation_fee_cents from category table, by category.
    -   Item-category mapping, authorization, lot/order bounds: from item tables (item_ref, category, authorized, minimum_lot, maximum_order).
    -   Resource usage per item: amount from usage tables, by (item_ref, resource).
    -   Resource total available: sum of amount for each resource in capacity_ledger table (sum all entries for each resource).
    -   Incompatibility: pairs from incompatible table.
    -   Prerequisites: pairs from requires table.
    -   Bundle bonuses: bonus_cents from bundle table, by (item_a, item_b).
6.  **Formulate Objective:** Maximize total net benefit in cents:
    -   Sum over all items: (per-item benefit) × x[i]
    -   Minus: sum over all items: (item activation fee) × y[i] (fee paid once per item if any quantity is chosen)
    -   Minus: sum over all categories: (category activation fee) × z[c] (fee paid once per category if any item in category is chosen)
    -   Plus: sum over all bundles: (bundle bonus) × b[bundle] (bonus only if both items in bundle are selected)
7.  **Formulate Constraints:**
    -   **Authorization and Quantity Bounds:** For each item i, x[i] = 0 if not authorized; otherwise, minimum_lot[i] ≤ x[i] ≤ maximum_order[i] or x[i] = 0.
    -   **Item Activation Linking:** For each item i, y[i] = 1 if x[i] ≥ 1, y[i] = 0 if x[i] = 0. (Enforce: x[i] ≥ y[i], x[i] ≤ maximum_order[i] × y[i])
    -   **Category Activation Linking:** For each category c, z[c] = 1 if any x[i] > 0 for i in c; z[c] = 0 otherwise. (Enforce: for all i in c, y[i] ≤ z[c]; and sum over i in c of y[i] ≥ z[c])
    -   **Category Quantity Bounds:** For each category c, sum over i in c of x[i] ≥ minimum_quantity[c] × z[c], and ≤ maximum_quantity[c] × z[c].
    -   **Resource Limits:** For each resource r, sum over all items i of (usage of r per unit of i) × x[i] ≤ total available of r (sum of capacity_ledger entries for r).
    -   **Incompatibility:** For each incompatible pair (i, j), y[i] + y[j] ≤ 1 (cannot both be selected).
    -   **Prerequisites:** For each (item, prerequisite) pair, y[item] ≤ y[prerequisite] (if item is selected, prerequisite must also be selected in positive quantity).
    -   **Bundle Bonuses:** For each bundle (i, j), b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1 (b[bundle] = 1 iff both y[i] = y[j] = 1).
    -   **Zero for Unauthorized:** For any item i not authorized, x[i] = 0, y[i] = 0.
    -   **Integrality:** All x[i] are integer, y[i], z[c], b[bundle] are binary.
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a portfolio of standardized development modules (indivisible options) in New York to maximize net return (benefit minus setup charges), subject to resource (storage, labor, energy) limits, category quantity bounds, item/category activation fees, incompatibility and prerequisite logic, and bundle bonuses. Only authorized options may be chosen, and each item must be selected in integer multiples within its allowed lot/order range.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, generalized assignment, and logical constraints (incompatibility, prerequisites, bundles).
3.  **Define Index Sets:** The primary indices are:
    - Items (item_ref): All item options explicitly listed in the two item tables (export_07.csv and export_08.csv), filtered to those with authorized=1.
    - Categories (category): As listed in the category table (export_04.csv).
    - Resources (resource): As listed in the usage and capacity_ledger tables (space, labor, power).
    - Bundles: Pairs of items from the bundle table (export_02.csv).
    - Incompatibles: Pairs of items from the incompatible table (export_06.csv).
    - Requires: Pairs of items from the requires table (export_11.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item i selected (0 if not selected, else between minimum_lot and maximum_order for authorized items). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is selected (category is "active"), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected (for bundle bonuses), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-item benefit: Sum of all amount_cents for each item_ref in the benefit table (export_01.csv).
    -   Per-item activation fee: activation_fee_cents from item_fee table (export_09.csv).
    -   Per-category activation fee: activation_fee_cents from category table (export_04.csv).
    -   Per-item resource usage: amount from usage tables (export_12.csv and export_13.csv), by item_ref and resource.
    -   Resource capacities: Sum of amount for each resource in capacity_ledger table (export_03.csv).
    -   Category quantity bounds: minimum_quantity and maximum_quantity from category table (export_04.csv).
    -   Item-category mapping, authorization, minimum_lot, maximum_order: from item tables (export_07.csv and export_08.csv).
    -   Bundle bonuses: bonus_cents from bundle table (export_02.csv).
    -   Incompatibility and requires logic: from incompatible (export_06.csv) and requires (export_11.csv) tables.
6.  **Formulate Objective:** Maximize total net benefit in cents:
    -   Sum over all items: (per-unit benefit * x[i]) 
    -   Minus: sum of item activation fees for each item selected (y[i])
    -   Minus: sum of category activation fees for each category used (z[c])
    -   Plus: sum of bundle bonuses for each bundle where both items are selected (b[bundle])
7.  **Formulate Constraints:**
    -   **Item selection and bounds:** For each authorized item i, x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i]; for unauthorized items, x[i] = 0.
    -   **Item activation indicator:** For each item i, y[i] = 1 if x[i] > 0, else 0. (Enforced via x[i] ≤ maximum_order[i] * y[i] and x[i] ≥ minimum_lot[i] * y[i])
    -   **Category activation indicator:** For each category c, z[c] = 1 if any item in c is selected (z[c] ≥ y[i] for all i in c; z[c] ≤ sum of y[i] over i in c).
    -   **Category quantity bounds:** For each category c, sum of x[i] over items i in c must be between minimum_quantity[c] and maximum_quantity[c] if z[c]=1, else zero.
    -   **Resource limits:** For each resource r, sum over all items of (usage of r per unit of i * x[i]) ≤ total available capacity for r (from capacity_ledger).
    -   **Incompatibility:** For each incompatible pair (i, j), y[i] + y[j] ≤ 1.
    -   **Requires:** For each (i, prereq), y[i] ≤ y[prereq].
    -   **Bundle bonuses:** For each bundle (i, j), b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1.
    -   **Integrality:** All x[i] are integer, y[i], z[c], b[bundle] are binary.
[Abstract Model Plan END]
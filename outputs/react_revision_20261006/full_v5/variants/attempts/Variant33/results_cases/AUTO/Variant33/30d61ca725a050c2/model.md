[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a portfolio of standardized development modules (indivisible options) in New York to maximize net return (benefit minus setup charges), subject to resource (storage, labor, energy) limits, category quantity bounds, item/category activation fees, incompatibility and prerequisite rules, and bundle bonuses. Only authorized options may be chosen, and each item must be selected in integer multiples within its allowed lot/order range.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, generalized assignment, and logical constraints (linking, incompatibility, prerequisites, bundles).
3.  **Define Index Sets:** The primary indices are:
    - Items (item_ref): All item_refs from the union of the two item tables (export_07.csv and export_08.csv), filtered to authorized=1.
    - Categories (category): All categories from the category table (export_04.csv).
    - Resources (resource): All resources from the capacity_ledger and usage tables (labor, space, power).
    - Bundles: All (item_a, item_b) pairs from the bundle table (export_02.csv).
    - Incompatibles: All (item_a, item_b) pairs from the incompatible table (export_06.csv).
    - Prerequisites: All (item_ref, prerequisite_ref) pairs from the requires table (export_11.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of units of item i selected (integer, 0 if not selected). Type: GRB.INTEGER, with bounds [0 or minimum_lot[i], maximum_order[i]] for authorized items; 0 for unauthorized.
    -   `y[i]` = 1 if item i is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is selected (i.e., category is "active"), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected (for bundle bonus), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each item: sum of amount_cents for each item_ref in the benefit table (export_01.csv).
    -   Item activation fee: activation_fee_cents from item_fee table (export_09.csv).
    -   Bundle bonus: bonus_cents from bundle table (export_02.csv).
    -   Resource usage per item: amount from usage tables (export_12.csv and export_13.csv), by item_ref and resource.
    -   Resource capacity: sum of amount for each resource in capacity_ledger table (export_03.csv).
    -   Category bounds and activation fee: minimum_quantity, maximum_quantity, activation_fee_cents from category table (export_04.csv).
    -   Item-category mapping, authorization, lot/order bounds: from item tables (export_07.csv and export_08.csv).
    -   Incompatibility pairs: from incompatible table (export_06.csv).
    -   Prerequisite pairs: from requires table (export_11.csv).
6.  **Formulate Objective:** Maximize total net benefit in cents:
    -   Sum over all items: (per-unit benefit[i] * x[i]) 
    -   Minus sum over all selected items: item activation_fee_cents[i] * y[i]
    -   Plus sum over all selected bundles: bonus_cents[bundle] * b[bundle]
    -   Minus sum over all active categories: activation_fee_cents[c] * z[c]
    -   (All terms in cents; report final value in USD cents as requested.)
7.  **Formulate Constraints:**
    -   **Item selection and bounds:** For each item i:
        - If authorized[i] = 0, x[i] = 0.
        - If authorized[i] = 1, x[i] = 0 or x[i] in [minimum_lot[i], maximum_order[i]] (integer).
        - y[i] = 1 if x[i] > 0, else 0 (enforced via x[i] >= minimum_lot[i] * y[i], x[i] <= maximum_order[i] * y[i]).
    -   **Resource limits:** For each resource r:
        - sum over all items i of (usage[i, r] * x[i]) ≤ total available capacity[r] (sum of capacity_ledger amounts for r).
    -   **Category quantity bounds:** For each category c:
        - sum over all items i in category c of x[i] ≥ minimum_quantity[c] * z[c]
        - sum over all items i in category c of x[i] ≤ maximum_quantity[c] * z[c]
        - z[c] = 1 if any x[i] > 0 for i in c, else 0.
    -   **Incompatibility:** For each incompatible pair (i, j):
        - y[i] + y[j] ≤ 1 (cannot select both).
    -   **Prerequisite:** For each (i, prereq) in requires:
        - y[i] ≤ y[prereq] (if i is selected, prereq must also be selected; no proportionality).
    -   **Bundle bonuses:** For each bundle (i, j):
        - b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1 (b[bundle]=1 iff both y[i]=y[j]=1).
        - If either i or j is unauthorized, b[bundle]=0.
    -   **Variable domains:** All x[i] integer, y[i], z[c], b[bundle] binary.
[Abstract Model Plan END]
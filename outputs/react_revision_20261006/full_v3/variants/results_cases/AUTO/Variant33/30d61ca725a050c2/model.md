[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a portfolio of standardized development modules (indivisible options) in New York to maximize net return (benefit minus setup charges), subject to resource (storage, labor, energy) limits, category quantity bounds, item/category activation fees, incompatibility and prerequisite rules, and bundle bonuses. Only authorized options may be chosen, and each item must be selected in integer multiples within its allowed lot/order range.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, generalized assignment, and logical constraints (incompatibility, prerequisites, bundles).
3.  **Define Index Sets:** The primary indices are:
    - Items (item_ref): All item_refs from the union of the two item tables (export_07.csv and export_08.csv), filtered to authorized=1.
    - Categories (category): All categories from the category table (export_04.csv).
    - Resources (resource): All resources from the capacity_ledger and usage tables (labor, space, power).
    - Bundles: All (item_a, item_b) pairs from the bundle table (export_02.csv).
    - Incompatibles: All (item_a, item_b) pairs from the incompatible table (export_06.csv).
    - Prerequisites: All (item_ref, prerequisite_ref) pairs from the requires table (export_11.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item i selected (0 if not selected). Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]] for authorized items; 0 for unauthorized.
    -   `y[i]` = 1 if item i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is selected (category is "active"), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-item benefit: Sum of amount_cents for each item_ref from the benefit table (export_01.csv).
    -   Per-item activation fee: activation_fee_cents from item_fee table (export_09.csv).
    -   Per-category activation fee: activation_fee_cents from category table (export_04.csv).
    -   Per-bundle bonus: bonus_cents from bundle table (export_02.csv).
    -   Resource usage per item: amount from usage tables (export_12.csv and export_13.csv), by item_ref and resource.
    -   Resource capacities: sum of amount for each resource from capacity_ledger table (export_03.csv).
    -   Category bounds: minimum_quantity and maximum_quantity from category table (export_04.csv).
    -   Item-category mapping: from item tables (export_07.csv and export_08.csv).
    -   Item authorization, minimum_lot, maximum_order: from item tables (export_07.csv and export_08.csv).
    -   Incompatibility and prerequisite pairs: from incompatible (export_06.csv) and requires (export_11.csv).
6.  **Formulate Objective:** Maximize total net benefit in cents:
    - Sum over all items: (per-item benefit) × x[i]
    - Minus: sum of item activation fees for each item selected (y[i]=1)
    - Minus: sum of category activation fees for each category used (z[c]=1)
    - Plus: sum of bundle bonuses for each bundle where both items are selected (b[bundle]=1)
    - Objective:  
      Maximize  
      `sum_i (benefit[i] * x[i]) - sum_i (item_fee[i] * y[i]) - sum_c (category_fee[c] * z[c]) + sum_bundle (bonus[bundle] * b[bundle])`
7.  **Formulate Constraints:**
    -   **Item selection domain:** For each item i:
        - If authorized[i]=0, x[i]=0.
        - If authorized[i]=1, x[i]=0 or x[i] ∈ [minimum_lot[i], maximum_order[i]] (integer).
    -   **Item activation linking:** For each item i:
        - y[i]=1 if x[i]>0, y[i]=0 if x[i]=0. (Enforced via: x[i] ≥ minimum_lot[i] * y[i], x[i] ≤ maximum_order[i] * y[i])
    -   **Category activation linking:** For each category c:
        - z[c]=1 if any x[i]>0 for i in c; z[c]=0 otherwise. (Enforced via: for all i in c, x[i] ≤ maximum_order[i] * z[c]; and sum_i_in_c x[i] ≥ minimum_lot[i] * z[c] if needed)
    -   **Category quantity bounds:** For each category c:
        - sum over i in c of x[i] ≥ minimum_quantity[c]
        - sum over i in c of x[i] ≤ maximum_quantity[c]
    -   **Resource constraints:** For each resource r:
        - sum over all items i of (usage[i,r] * x[i]) ≤ total available capacity[r] (sum of capacity_ledger amounts for r)
    -   **Incompatibility constraints:** For each incompatible pair (i,j):
        - y[i] + y[j] ≤ 1 (cannot select both)
    -   **Prerequisite constraints:** For each (i, prereq):
        - y[i] ≤ y[prereq] (if i is selected, prereq must be selected)
    -   **Bundle bonus linking:** For each bundle (i,j):
        - b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1 (b[bundle]=1 iff both y[i]=1 and y[j]=1)
    -   **Variable domains:** All x[i] integer, y[i], z[c], b[bundle] binary.
[Abstract Model Plan END]
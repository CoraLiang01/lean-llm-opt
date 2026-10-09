[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a portfolio of standardized development modules (indivisible options) in New York to maximize net return (benefit minus setup charges), subject to resource (storage, labor, energy) limits, category quantity bounds, item/category activation fees, incompatibility and prerequisite logic, and bundle bonuses. Only authorized options may be chosen, and each item must be selected in integer multiples within its allowed lot/order range.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, generalized assignment, and logical constraints (incompatibility, prerequisites, bundles).
3.  **Define Index Sets:** The primary indices are:
    - Items (item_ref): All item options explicitly listed in the item tables (from both batch_01/export_07.csv and batch_02/export_08.csv).
    - Categories (category): As listed in the category table (batch_04/export_04.csv).
    - Resources (resource): As listed in the capacity_ledger and usage tables (labor, space, power).
    - Bundles: Pairs of items from the bundle table (batch_02/export_02.csv).
    - Incompatibility pairs: From the incompatible table (batch_06/export_06.csv).
    - Prerequisite pairs: From the requires table (batch_05/export_11.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item `i` selected (0 if not selected). Type: GRB.INTEGER. Domain: 0 or any integer in [minimum_lot[i], maximum_order[i]] if authorized, else 0.
    -   `y[i]` = 1 if item `i` is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category `c` is selected (i.e., category is "active"), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected (for bundle bonus), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-item benefit: Sum of all `amount_cents` for each item_ref in the benefit table (batch_01/export_01.csv).
    -   Per-item activation fee: `activation_fee_cents` from item_fee table (batch_03/export_09.csv).
    -   Per-category activation fee: `activation_fee_cents` from category table (batch_04/export_04.csv).
    -   Per-item resource usage: `amount` for each (item_ref, resource) from usage tables (batch_06/export_12.csv and batch_01/export_13.csv).
    -   Resource capacities: Sum of `amount` for each resource in capacity_ledger table (batch_03/export_03.csv).
    -   Item authorization, minimum_lot, maximum_order, and category: from item tables (batch_01/export_07.csv and batch_02/export_08.csv).
    -   Category quantity bounds: minimum_quantity, maximum_quantity from category table (batch_04/export_04.csv).
    -   Bundle bonuses: `bonus_cents` from bundle table (batch_02/export_02.csv).
    -   Incompatibility pairs: from incompatible table (batch_06/export_06.csv).
    -   Prerequisite pairs: from requires table (batch_05/export_11.csv).
6.  **Formulate Objective:** Maximize total net benefit in cents:
    -   Sum over all items: (per-item benefit) × x[i]
    -   Minus: sum of item activation fees for each item selected (i.e., y[i]=1)
    -   Minus: sum of category activation fees for each category used (z[c]=1)
    -   Plus: sum of bundle bonuses for each bundle where both items are selected (b[bundle]=1)
    -   (All terms in cents; report final value in USD cents as requested.)
7.  **Formulate Constraints:**
    -   **Item selection domain:** For each item i:
        - If authorized[i]=0, x[i]=0.
        - If authorized[i]=1, x[i]=0 or x[i] in [minimum_lot[i], maximum_order[i]] (integer).
    -   **Item activation linking:** For each item i:
        - y[i]=1 if x[i]>0, y[i]=0 if x[i]=0. (Enforced via: x[i] ≥ minimum_lot[i] * y[i], x[i] ≤ maximum_order[i] * y[i])
    -   **Category activation linking:** For each category c:
        - z[c]=1 if any x[i]>0 for item i in category c; z[c]=0 otherwise. (Enforced via: for all i in c, x[i] ≤ maximum_order[i] * z[c]; and sum over i in c of y[i] ≥ z[c])
    -   **Category quantity bounds:** For each category c:
        - sum over i in c of x[i] ≥ minimum_quantity[c]
        - sum over i in c of x[i] ≤ maximum_quantity[c]
    -   **Resource capacity constraints:** For each resource r:
        - sum over all items i of (usage[i,r] × x[i]) ≤ total available capacity[r] (sum of opening and reservation in capacity_ledger)
    -   **Incompatibility constraints:** For each incompatible pair (i,j):
        - y[i] + y[j] ≤ 1 (cannot select both)
    -   **Prerequisite constraints:** For each (i, prerequisite):
        - y[i] ≤ y[prerequisite] (if i is selected, prerequisite must also be selected)
    -   **Bundle bonus linking:** For each bundle (i,j):
        - b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1 (b[bundle]=1 iff both y[i]=1 and y[j]=1)
    -   **Variable domains:** All x[i] integer, y[i], z[c], b[bundle] binary.
[Abstract Model Plan END]
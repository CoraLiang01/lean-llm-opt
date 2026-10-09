[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate licensed game-edition packages (item/platform options) to PC, CONSOLE, and MOBILE, maximizing net licensing return (benefit after fees and bonuses), subject to per-platform memory capacity, per-category quantity bounds, item/category activation fees, authorization, minimum/maximum order sizes, incompatibility and prerequisite logic, and bundle bonuses. All data is to be taken directly from the supplied tables.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items/options: all rows in the item table (item_ref).
    - Platforms: location_id (PC, CONSOLE, MOBILE), as given in the item table.
    - Categories: all rows in the category table (category).
    - Resources: all rows in the capacity_ledger and usage tables (resource: PC, CONSOLE, MOBILE).
    - Bundles: all rows in the bundle table (pairs of item_refs).
4.  **Define Decision Variables:**
    -   `x[i]` = integer quantity of item_ref `i` to allocate (0 or in [minimum_lot, maximum_order] if authorized; 0 if unauthorized). Type: GRB.INTEGER.
    -   `y[i]` = binary indicator if item_ref `i` is selected (1 if x[i] > 0, 0 otherwise). Type: GRB.BINARY.
    -   `z[g]` = binary indicator if category `g` is activated (1 if any item in category g is selected, 0 otherwise). Type: GRB.BINARY.
    -   `b[a,b]` = binary indicator if both items in bundle (a,b) are selected (1 if y[a]=1 and y[b]=1, 0 otherwise). Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit per unit for each item: sum of all benefit table rows for item_ref, each converted to USD cents using fx table (amount * usd_cents_numerator / denominator).
    -   Item activation fee: item_fee table, activation_fee_cents per item_ref.
    -   Category activation fee: category table, activation_fee_cents per category.
    -   Bundle bonus: bundle table, bonus_cents per (item_a, item_b) pair.
    -   Memory usage per unit: usage table, amount (convert GB to MB using 1000 MB/GB) per item_ref and resource.
    -   Platform memory capacity: sum of capacity_ledger table rows per resource (opening + reservation), in MB.
    -   Category bounds: category table, minimum_quantity and maximum_quantity per category.
    -   Authorization, minimum_lot, maximum_order, category, location_id: item table, per item_ref.
    -   Incompatibility: incompatible table, pairs of item_refs that cannot both be selected.
    -   Prerequisite: requires table, item_ref and prerequisite_ref pairs (if item_ref selected, prerequisite_ref must be selected).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (benefit per unit in USD cents) * x[i]
    -   Minus: sum of item activation_fee_cents for each item_ref with y[i]=1
    -   Minus: sum of category activation_fee_cents for each category with z[g]=1
    -   Plus: sum of bundle bonus_cents for each bundle (a,b) with b[a,b]=1
7.  **Formulate Constraints:**
    -   **Authorization and Order Bounds:** For each item_ref:
        - If authorized=0, x[i]=0.
        - If authorized=1, x[i]=0 or x[i] in [minimum_lot, maximum_order].
        - y[i]=1 iff x[i]>0; y[i]=0 iff x[i]=0.
    -   **Platform Memory Capacity:** For each resource (platform):
        - Sum over all items assigned to that resource: (usage per unit in MB) * x[i] ≤ total available capacity (sum of capacity_ledger entries for that resource).
    -   **Category Quantity Bounds:** For each category:
        - Sum of x[i] over all items in the category ≥ minimum_quantity.
        - Sum of x[i] over all items in the category ≤ maximum_quantity.
        - z[g]=1 iff any y[i]=1 for items in category g; z[g]=0 otherwise.
    -   **Item Activation Fee Logic:** For each item_ref:
        - Deduct activation_fee_cents only if y[i]=1 (enforced via variable and objective).
    -   **Category Activation Fee Logic:** For each category:
        - Deduct activation_fee_cents only if z[g]=1 (enforced via variable and objective).
    -   **Incompatibility:** For each incompatible pair (a,b):
        - y[a] + y[b] ≤ 1 (cannot both be selected).
    -   **Prerequisite:** For each requires pair (i,prereq):
        - y[i] ≤ y[prereq] (if i is selected, prereq must be selected).
    -   **Bundle Bonus Logic:** For each bundle (a,b):
        - b[a,b] ≤ y[a], b[a,b] ≤ y[b], b[a,b] ≥ y[a] + y[b] - 1 (b[a,b]=1 iff both y[a]=1 and y[b]=1).
    -   **Variable Domains:** x[i] integer, y[i], z[g], b[a,b] binary.
[Abstract Model Plan END]
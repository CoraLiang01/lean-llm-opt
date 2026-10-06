[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select integer quantities of authorized vehicle configurations for RIVERSIDE_AUTO’s next delivery, maximizing total net benefit (unit benefit minus fixed preparation charges), subject to resource, category, compatibility, dependency, and bundle bonus rules. All data is to be taken directly from the supplied tables.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, category constraints, logical (incompatibility and dependency) constraints, and bundle bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items/configurations (`i`): All rows from the item table (export_06.csv) with their unique `item_ref`.
    - Categories (`g`): All rows from the category table (export_03.csv) with their unique `category`.
    - Resources (`r`): All resources from the usage and capacity_ledger tables (e.g., power, labor, space).
    - Bundles (`(a,b)`): All pairs from the bundle table (export_01.csv).
    - Incompatible pairs (`(a,b)`): All pairs from the incompatible table (export_05.csv).
    - Requires pairs (`(i,prereq)`): All pairs from the requires table (export_08.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item/configuration `i` to order. Type: GRB.INTEGER. Must be zero if not authorized.
    -   `y[i]` = Binary variable: 1 if any of item `i` is ordered (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = Binary variable: 1 if any item in category `g` is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `b[a,b]` = Binary variable: 1 if both items `a` and `b` are ordered (i.e., x[a]>0 and x[b]>0), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents` (from item table, export_06.csv): per-unit benefit for each item.
        -   `item_fee_cents` (from item table): fixed fee per item if any are ordered.
        -   `activation_fee_cents` (from category table, export_03.csv): fixed fee per category if any item in the category is ordered.
        -   `bonus_cents` (from bundle table, export_01.csv): bonus for each bundle if both items are ordered.
    -   Constraint coefficients:
        -   `usage` (from usage table, export_09.csv): per-unit resource usage for each item and resource.
        -   `capacity_ledger` (from capacity_ledger table, export_02.csv): total available capacity for each resource (sum opening and reservation for each resource).
        -   `minimum_lot`, `maximum_order`, `authorized` (from item table): lot size and order bounds, and authorization status for each item.
        -   `minimum_quantity`, `maximum_quantity` (from category table): category-level quantity bounds.
        -   Incompatible pairs (from incompatible table, export_05.csv): pairs of items that cannot both be ordered.
        -   Requires pairs (from requires table, export_08.csv): item dependencies.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over all items: (item_fee_cents[i] * y[i]) [charged once per item if any ordered]
    -   Minus sum over all categories: (activation_fee_cents[g] * z[g]) [charged once per category if any item in category is ordered]
    -   Plus sum over all bundles: (bonus_cents[a,b] * b[a,b]) [awarded once per bundle if both items are ordered]
7.  **Formulate Constraints:**
    -   **Authorization and Order Bounds:** For each item `i`, x[i] = 0 if authorized[i] = 0; otherwise, minimum_lot[i] ≤ x[i] ≤ maximum_order[i] or x[i] = 0.
    -   **Item Activation Linking:** For each item `i`, y[i] = 1 if x[i] > 0, else 0. (Enforced via x[i] ≤ maximum_order[i] * y[i], x[i] ≥ minimum_lot[i] * y[i], y[i] ∈ {0,1}).
    -   **Category Activation Linking:** For each category `g`, z[g] = 1 if any x[i] > 0 for i in category g, else 0. (Enforced via x[i] ≤ M * z[g] for all i in g, and z[g] ≥ y[i] for all i in g).
    -   **Category Quantity Bounds:** For each category `g`, minimum_quantity[g] ≤ sum_{i in g} x[i] ≤ maximum_quantity[g].
    -   **Resource Capacity Constraints:** For each resource `r`, sum over all items: (usage[i,r] * x[i]) ≤ total available capacity for resource r (sum of opening and reservation for r in capacity_ledger).
    -   **Incompatibility Constraints:** For each incompatible pair (a,b), y[a] + y[b] ≤ 1 (cannot both be ordered).
    -   **Requires Constraints:** For each requires pair (i,prereq), y[i] ≤ y[prereq] (cannot order i unless prerequisite is also ordered).
    -   **Bundle Bonus Linking:** For each bundle (a,b), b[a,b] = 1 if y[a] = 1 and y[b] = 1, else 0. (Enforced via b[a,b] ≤ y[a], b[a,b] ≤ y[b], b[a,b] ≥ y[a] + y[b] - 1).
    -   **Variable Domains:** x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer), y[i], z[g], b[a,b] ∈ {0,1}.
[Abstract Model Plan END]
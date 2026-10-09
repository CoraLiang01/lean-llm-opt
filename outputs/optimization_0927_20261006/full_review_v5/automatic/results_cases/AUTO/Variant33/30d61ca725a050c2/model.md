[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a portfolio of authorized, indivisible development modules (items) in New York to maximize net return (benefit minus setup charges), subject to resource (storage, labor, energy) limits, category quantity bounds, item/category activation fees, incompatibility and prerequisite logic, and bundle bonuses. All constraints and objective terms are to be modeled exactly as described, using the supplied tables and their rows directly.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee), generalized assignment, and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (`i`): All item_refs from the union of item tables (from both batch_01/export_07.csv and batch_02/export_08.csv), filtered to authorized items only.
    - Categories (`g`): All categories from the category table (batch_04/export_04.csv).
    - Resources (`r`): All resources from the capacity_ledger and usage tables (labor, space, power).
    - Bundles (`(i,j)`): All item pairs from the bundle table (batch_02/export_02.csv).
    - Incompatible pairs (`(i,j)`): All item pairs from the incompatible table (batch_06/export_06.csv).
    - Requires pairs (`(i,j)`): All item-prerequisite pairs from the requires table (batch_05/export_11.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item `i` selected (0 or an integer in [minimum_lot[i], maximum_order[i]] if selected, 0 if not selected). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item `i` is selected (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category `g` is selected (i.e., sum_{i in g} y[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[i,j]` = 1 if both items `i` and `j` are selected (for each bundle pair), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-item benefit: Sum of all amount_cents for each item_ref in the benefit table (batch_01/export_01.csv).
    -   Per-item activation fee: activation_fee_cents from the item_fee table (batch_03/export_09.csv).
    -   Per-category activation fee: activation_fee_cents from the category table (batch_04/export_04.csv).
    -   Per-item resource usage: amount from the usage tables (batch_06/export_12.csv and batch_01/export_13.csv), by item_ref and resource.
    -   Resource capacities: Sum of amount for each resource in the capacity_ledger table (batch_03/export_03.csv).
    -   Item-category mapping, minimum_lot, maximum_order, and authorization: from the item tables (batch_01/export_07.csv and batch_02/export_08.csv).
    -   Bundle bonuses: bonus_cents from the bundle table (batch_02/export_02.csv).
    -   Incompatibility and requires relations: from the incompatible (batch_06/export_06.csv) and requires (batch_05/export_11.csv) tables.
    -   Category quantity bounds: minimum_quantity and maximum_quantity from the category table (batch_04/export_04.csv).
6.  **Formulate Objective:** Maximize total net benefit in cents:
        - Sum over all items: (per-unit benefit[i] * x[i]) 
        - Minus sum over all selected items: (item activation_fee_cents[i] * y[i])
        - Minus sum over all used categories: (category activation_fee_cents[g] * z[g])
        - Plus sum over all bundle pairs: (bonus_cents[i,j] * b[i,j])
7.  **Formulate Constraints:**
    -   **Item selection and bounds:** For each authorized item `i`, enforce: x[i] = 0 or x[i] in [minimum_lot[i], maximum_order[i]]; i.e., minimum_lot[i] * y[i] ≤ x[i] ≤ maximum_order[i] * y[i], and x[i] = 0 if y[i] = 0.
    -   **Authorization:** For each item, if authorized = 0, force x[i] = 0 and y[i] = 0.
    -   **Resource limits:** For each resource `r`, sum over all items: sum_i (usage[i,r] * x[i]) ≤ total capacity[r] (sum of capacity_ledger amounts for resource r).
    -   **Category quantity bounds:** For each category `g`, sum over all items in category: minimum_quantity[g] ≤ sum_{i in g} x[i] ≤ maximum_quantity[g].
    -   **Category activation flag:** For each category `g`, z[g] = 1 if any y[i] = 1 for i in g; i.e., y[i] ≤ z[g] for all i in g, and z[g] ≤ sum_{i in g} y[i].
    -   **Incompatibility:** For each incompatible pair (i,j), y[i] + y[j] ≤ 1.
    -   **Requires:** For each requires pair (i,j), y[i] ≤ y[j] (i requires j).
    -   **Bundle logic:** For each bundle pair (i,j), b[i,j] ≤ y[i], b[i,j] ≤ y[j], and b[i,j] ≥ y[i] + y[j] - 1 (b[i,j] = 1 iff both y[i] = y[j] = 1).
[Abstract Model Plan END]
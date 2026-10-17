[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select integer quantities of authorized vehicle configurations for delivery, maximizing total net benefit (unit benefit per item, minus fixed item and category fees, plus bundle bonuses), subject to resource capacity, category quantity bounds, item eligibility, incompatibility, and dependency (requires) constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, and logical (compatibility/dependency) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (from the 'item' table, filtered to authorized options)
    - Categories (from the 'category' table)
    - Resources (from the 'capacity_ledger' and 'usage' tables)
    - Bundles (from the 'bundle' table: pairs of items eligible for a bonus)
    - Incompatible pairs (from the 'incompatible' table)
    - Requires pairs (from the 'requires' table)
4.  **Define Decision Variables:**
    - `q[i]` = Integer quantity of item `i` selected for delivery (0 or in [minimum_lot, maximum_order] if authorized; 0 if unauthorized). Type: GRB.INTEGER.
    - `z[i]` = Binary flag: 1 if item `i` is selected (i.e., q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    - `y[c]` = Binary flag: 1 if any item in category `c` is selected, 0 otherwise. Type: GRB.BINARY.
    - `b[bundle]` = Binary flag: 1 if both items in bundle are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - `unit_benefit_cents` (from 'item' table): per-unit benefit for each item.
        - `item_fee_cents` (from 'item' table): fixed fee per item if any quantity is selected.
        - `activation_fee_cents` (from 'category' table): fixed fee per category if any item in the category is selected.
        - `bonus_cents` (from 'bundle' table): bonus for each bundle if both items are selected.
    - Constraint coefficients:
        - `amount` (from 'usage' table): per-unit resource usage for each item and resource.
        - `amount` (from 'capacity_ledger' table): total available (sum of 'opening' and 'reservation') for each resource.
        - `minimum_lot`, `maximum_order` (from 'item' table): per-item quantity bounds.
        - `minimum_quantity`, `maximum_quantity` (from 'category' table): per-category total quantity bounds.
    - Logical constraints:
        - `authorized` (from 'item' table): eligibility for selection.
        - Incompatible pairs (from 'incompatible' table): pairs of items that cannot both be selected.
        - Requires pairs (from 'requires' table): item dependencies (if item is selected, prerequisite must also be selected).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over items: (unit_benefit_cents[i] * q[i]) 
    - Minus sum over items: (item_fee_cents[i] * z[i]) [charged once per item if any quantity is selected]
    - Minus sum over categories: (activation_fee_cents[c] * y[c]) [charged once per category if any item in the category is selected]
    - Plus sum over bundles: (bonus_cents[bundle] * b[bundle]) [awarded once per bundle if both items are selected]
7.  **Formulate Constraints:**
    - **Item eligibility and quantity bounds:** For each item:
        - If authorized: q[i] = 0 or q[i] in [minimum_lot[i], maximum_order[i]]; enforce q[i] >= minimum_lot[i] * z[i], q[i] <= maximum_order[i] * z[i], and q[i] = 0 if z[i] = 0.
        - If unauthorized: q[i] = 0, z[i] = 0.
    - **Resource capacity constraints:** For each resource:
        - Sum over items: (usage amount per unit for resource r and item i) * q[i] ≤ total available for resource r (sum of 'opening' and 'reservation' in 'capacity_ledger' for r).
    - **Category quantity bounds:** For each category:
        - Sum over items in category c: q[i] ≥ minimum_quantity[c]
        - Sum over items in category c: q[i] ≤ maximum_quantity[c]
        - y[c] ≥ z[i] for all items i in category c; y[c] = 1 iff any item in c is selected.
    - **Incompatibility constraints:** For each incompatible pair (i, j):
        - z[i] + z[j] ≤ 1 (at most one of the pair can be selected).
    - **Requires (dependency) constraints:** For each requires pair (i, prereq):
        - z[i] ≤ z[prereq] (if i is selected, its prerequisite must also be selected).
    - **Bundle bonus activation:** For each bundle (i, j):
        - b[bundle] ≤ z[i], b[bundle] ≤ z[j], b[bundle] ≥ z[i] + z[j] - 1 (b[bundle] = 1 iff both items are selected).
[Abstract Model Plan END]
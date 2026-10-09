[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine, as of 2026-05-07, the vehicle order for the Oslo dealership that yields the largest net benefit in USD cents, subject to multi-currency benefit components, item and category fees, resource and category limits, option authorization, incompatibility and dependency rules, and bundle bonuses. All data must be filtered per a strict as-of/revision/deletion rule for each table before use.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, resource constraints, logical dependencies, and combinatorial (bundle/incompatibility) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items/options (`I`): vehicle options/items available for order.
    - Categories (`C`): groups of items with aggregate quantity limits and activation fees.
    - Resources (`R`): resource types (e.g., space, labor, power) with capacity limits.
    - Bundles (`B`): pairs of items eligible for a bundle bonus.
    - Incompatible pairs (`P`): pairs of items that cannot both be selected.
    - Requires pairs (`Q`): pairs where one item requires the other.
4.  **Define Decision Variables:**
    - `q[i]` = integer quantity ordered of item/option `i` (within [minimum_lot, maximum_order] if authorized, else zero). Type: GRB.INTEGER.
    - `z[i]` = binary indicator if item `i` is selected (i.e., q[i] > 0). Type: GRB.BINARY.
    - `y[c]` = binary indicator if any item in category `c` is selected (category used). Type: GRB.BINARY.
    - `w[b]` = binary indicator if both items in bundle `b` are selected (bundle awarded). Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Benefit components: from `benefit` table, sum of all signed amounts per item, converted to USD cents using `fx` table (`amount * usd_cents_numerator / denominator`).
    - Item fees: from `item_fee` table, `activation_fee_cents` per item, charged once if any quantity of that item is ordered.
    - Category limits and activation fees: from `category` table, `minimum_quantity`, `maximum_quantity`, `activation_fee_cents` per category.
    - Item authorization, lot/order bounds, and category mapping: from `item` table, `authorized`, `minimum_lot`, `maximum_order`, `category`.
    - Resource usage per unit: from `usage` table, `amount` and `unit` per item-resource pair (convert all to base units: 1000 ml/liter, 60 min/hour, 1000 wh/kwh).
    - Resource capacities: from `capacity_ledger` table, sum of all signed `amount` per resource (in base units).
    - Incompatibility: from `incompatible` table, pairs of items that cannot both be selected.
    - Requires: from `requires` table, pairs where one item requires another (positive quantity).
    - Bundle bonuses: from `bundle` table, `bonus_cents` per eligible item pair.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over all items: (total per-unit benefit in USD cents) × q[i]
    - Minus: sum of item fees for each item with q[i] > 0
    - Minus: sum of category activation fees for each category with at least one item selected
    - Plus: sum of bundle bonuses for each bundle where both items are selected
7.  **Formulate Constraints:**
    - **Item selection and bounds:** For each item i:
        - If authorized: minimum_lot[i] × z[i] ≤ q[i] ≤ maximum_order[i] × z[i]; q[i] ≥ 0 integer; z[i] ∈ {0,1}
        - If not authorized: q[i] = 0; z[i] = 0
    - **Category quantity limits:** For each category c: minimum_quantity[c] × y[c] ≤ sum_{i in c} q[i] ≤ maximum_quantity[c] × y[c]; y[c] ∈ {0,1}; z[i] ≤ y[c] for all i in c
    - **Resource capacity:** For each resource r: sum_{i} (resource_usage[i,r] × q[i]) ≤ capacity[r] (all in base units)
    - **Incompatibility:** For each incompatible pair (i,j): z[i] + z[j] ≤ 1
    - **Requires dependencies:** For each requires pair (i,j): q[i] ≤ maximum_order[i] × z[j]; q[j] ≥ 1 if q[i] ≥ 1
    - **Bundle bonuses:** For each bundle (i,j): w[b] ≤ z[i]; w[b] ≤ z[j]; w[b] ≥ z[i] + z[j] - 1; w[b] ∈ {0,1}
    - **Item fees:** For each item i: item fee charged once if z[i] = 1
    - **Category activation fees:** For each category c: activation fee charged once if y[c] = 1
    - **All variables:** Domains as above; all constraints apply only to valid, as-of-2026-05-07, non-deleted, highest-revision records per table/entity key, per the query’s selection rule.
[Abstract Model Plan END]
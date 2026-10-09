[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer replenishment quantities for each authorized option (item) for business unit NORTH, as of 2026-03-12, to maximize net benefit (in USD cents). The model must account for per-unit benefits, item and category activation fees, resource and category quantity limits, option compatibility and dependency rules, bundle bonuses, and unit conversions for resource usage and capacity. Only the latest non-future, non-DELETE revision per (tenant, table, record_id) is used from each table before joining.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, compatibility, and dependency constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (options) `i` (from item tables, filtered for NORTH and latest valid revision)
    - Categories `g` (from category tables, filtered for NORTH and latest valid revision)
    - Resources `r` (from resource usage and capacity tables, filtered for NORTH and latest valid revision)
    - Bundles `b` (from bundle tables, filtered for NORTH and latest valid revision)
    - Incompatible pairs `(i, j)` and requires pairs `(i, j)` (from respective tables, filtered as above)
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity ordered of item `i`. Type: GRB.INTEGER.
    -   `z[i]` = 1 if item `i` is selected (i.e., `x[i] > 0`), 0 otherwise. Type: GRB.BINARY.
    -   `w[g]` = 1 if any item in category `g` is selected, 0 otherwise. Type: GRB.BINARY.
    -   `u[b]` = 1 if both items in bundle `b` are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit: sum of `amount_cents` from 'benefit' tables for each item (filtered as above).
    -   Item activation fee: `activation_fee_cents` from 'item_fee' tables for each item.
    -   Category activation fee: `activation_fee_cents` from 'category' tables for each category.
    -   Bundle bonus: `bonus_cents` from 'bundle' tables for each bundle.
    -   Resource usage per unit: `amount` and `unit` from 'usage' tables for each item-resource pair.
    -   Resource capacity: sum of `amount` (with sign) and `unit` from 'capacity_ledger' tables for each resource.
    -   Item bounds: `minimum_lot`, `maximum_order`, and `authorized` from 'item' tables for each item.
    -   Category quantity bounds: `minimum_quantity`, `maximum_quantity` from 'category' tables for each category.
    -   Incompatible pairs: from 'incompatible' tables.
    -   Requires pairs: from 'requires' tables.
    -   All parameters filtered for tenant = NORTH, effective_date ≤ 2026-03-12, and latest revision not marked DELETE.
    -   Unit conversions: 1000 ml = 1 liter, 60 minutes = 1 hour, 1000 wh = 1 kwh.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over items: (per-unit benefit) × `x[i]`
    -   Minus: sum over items: (item activation fee) × `z[i]` (only if `x[i] > 0`)
    -   Minus: sum over categories: (category activation fee) × `w[g]` (only if any item in category is selected)
    -   Plus: sum over bundles: (bundle bonus) × `u[b]` (only if both items in bundle are selected)
7.  **Formulate Constraints:**
    -   **Item Authorization and Bounds:** For each item `i`, if `authorized` > 0, then `minimum_lot[i] ≤ x[i] ≤ maximum_order[i]`; if `authorized` = 0, then `x[i] = 0`.
    -   **Item Selection Indicator:** For each item `i`, `z[i] = 1` if `x[i] > 0`, else `z[i] = 0`.
    -   **Category Quantity Bounds:** For each category `g`, sum of `x[i]` over items in `g` satisfies `minimum_quantity[g] ≤ sum_i_in_g x[i] ≤ maximum_quantity[g]`.
    -   **Category Activation Indicator:** For each category `g`, `w[g] = 1` if any `x[i] > 0` for `i` in `g`, else `w[g] = 0`.
    -   **Resource Capacity:** For each resource `r`, sum over items of (resource usage per unit, converted to capacity units) × `x[i]` ≤ total available capacity for `r` (converted to same units).
    -   **Incompatibility:** For each incompatible pair `(i, j)`, at most one of `x[i]`, `x[j]` is positive: `z[i] + z[j] ≤ 1`.
    -   **Requires Dependency:** For each requires pair `(i, j)`, `x[i] > 0` ⇒ `x[j] > 0` (i.e., `z[i] ≤ z[j]`).
    -   **Bundle Bonus Indicator:** For each bundle `b` with items `(i, j)`, `u[b] = 1` if both `z[i] = 1` and `z[j] = 1`, else `u[b] = 0`.
    -   **Integrality:** All `x[i]` are integer, all indicator variables are binary.
[Abstract Model Plan END]
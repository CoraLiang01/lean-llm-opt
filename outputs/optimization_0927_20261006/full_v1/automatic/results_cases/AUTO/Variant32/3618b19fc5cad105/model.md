[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine, for the Oslo dealership as of 2026-05-07, the vehicle order (integer lot sizes for authorized options, subject to all operational, resource, compatibility, dependency, and fee rules) that yields the largest net benefit in USD cents, after all conversions, bonuses, and deductions.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, resource constraints, compatibility, and logical dependencies.
3.  **Define Index Sets:** The primary indices are:
    - Items/options (`i`): All valid OSLO_NEW_CARS items/options as of 2026-05-07.
    - Categories (`g`): All valid OSLO_NEW_CARS categories as of 2026-05-07.
    - Resources (`r`): All valid OSLO_NEW_CARS resources (e.g., space, power, labor).
    - Bundles (`b`): All valid OSLO_NEW_CARS bundle bonus pairs as of 2026-05-07.
    - Incompatible pairs (`(i,j)`): All valid OSLO_NEW_CARS incompatible item pairs as of 2026-05-07.
    - Requires pairs (`(i,k)`): All valid OSLO_NEW_CARS requires (dependency) pairs as of 2026-05-07.
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity ordered of option/item `i`. Type: GRB.INTEGER. Must be zero if unauthorized.
    -   `z[i]` = Binary variable: 1 if option/item `i` is selected (i.e., `q[i] > 0`), 0 otherwise. Type: GRB.BINARY.
    -   `y[g]` = Binary variable: 1 if any item in category `g` is selected (i.e., sum over `i` in `g` of `q[i] > 0`), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = Binary variable: 1 if both items in bundle `b` are selected (i.e., both corresponding `z[i]` are 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit components per item (in various currencies): from `benefit` table, columns `amount`, `currency`, `item_ref`, `component`.
    -   FX rates: from `fx` table, columns `currency`, `usd_cents_numerator`, `denominator`.
    -   Item fees: from `item_fee` table, columns `item_ref`, `activation_fee_cents`.
    -   Category limits and activation fees: from `category` table, columns `category`, `minimum_quantity`, `maximum_quantity`, `activation_fee_cents`.
    -   Item authorization, lot sizes, and order limits: from `item` table, columns `item_ref`, `authorized`, `minimum_lot`, `maximum_order`, `category`.
    -   Resource usage per unit: from `usage` table, columns `item_ref`, `resource`, `amount`, `unit`.
    -   Resource capacities: from `capacity_ledger` table, columns `resource`, `amount`, `unit`, summed over all valid entries.
    -   Bundle bonuses: from `bundle` table, columns `item_a`, `item_b`, `bonus_cents`.
    -   Incompatible pairs: from `incompatible` table, columns `item_a`, `item_b`.
    -   Requires dependencies: from `requires` table, columns `item_ref`, `prerequisite_ref`.
6.  **Formulate Objective:** Maximize total net benefit in USD cents, calculated as:
    -   For each item, sum all benefit components (after currency conversion) per unit, multiply by `q[i]`.
    -   Subtract the item fee (once per item with `q[i] > 0`).
    -   Subtract the category activation fee (once per category with any item selected).
    -   Add bundle bonuses (once per bundle where both items are selected).
7.  **Formulate Constraints:**
    -   **Authorization and Lot Constraints:** For each item `i`, `q[i] = 0` if not authorized; if authorized, `q[i]` is integer, and `minimum_lot[i] <= q[i] <= maximum_order[i]` or `q[i] = 0`.
    -   **Category Quantity Limits:** For each category `g`, sum of `q[i]` over items in `g` must satisfy `minimum_quantity[g] <= sum(q[i] for i in g) <= maximum_quantity[g]`.
    -   **Resource Capacity Constraints:** For each resource `r`, total usage across all items (converted to the resource's base unit: 1000 ml/liter, 60 min/hour, 1000 wh/kwh) must not exceed the signed total capacity from `capacity_ledger`.
    -   **Item Fee Application:** For each item, item fee is charged once if `q[i] > 0`.
    -   **Category Activation Fee:** For each category, activation fee is charged once if any `q[i] > 0` for items in `g`.
    -   **Bundle Bonus Application:** For each bundle, bonus is awarded once if both items are selected, zero otherwise.
    -   **Incompatibility Constraints:** For each incompatible pair `(i,j)`, at most one of `q[i]`, `q[j]` can be positive: `z[i] + z[j] <= 1`.
    -   **Requires Constraints:** For each requires pair `(i,k)`, if `q[i] > 0` then `q[k] > 0` (prerequisite must be present in positive quantity).
    -   **Linking Constraints:** For each item, `z[i] = 1` if `q[i] > 0`, else `z[i] = 0`; for each category, `y[g] = 1` if any `q[i] > 0` for items in `g`, else `y[g] = 0`; for each bundle, `w[b] = 1` if both items are selected, else `w[b] = 0`.
    -   **Integrality:** All `q[i]` are integer, all `z[i]`, `y[g]`, `w[b]` are binary.
[Abstract Model Plan END]
[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine, as of 2026-05-07, the vehicle order for the Oslo dealership (OSLO_NEW_CARS) that yields the largest net benefit in USD cents, subject to a complex set of eligibility, resource, category, compatibility, dependency, and fee/bonus rules. All data must be filtered by a precise as-of/revision/deletion rule for each logical table before use.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, resource constraints, compatibility and dependency logic, and multi-currency benefit aggregation.
3.  **Define Index Sets:** The primary indices are:
    - Items/options (`i`): All valid item_ref for OSLO_NEW_CARS as of 2026-05-07.
    - Categories (`g`): All valid categories for OSLO_NEW_CARS as of 2026-05-07.
    - Resources (`r`): All valid resources (e.g., space, labor, power) for OSLO_NEW_CARS as of 2026-05-07.
    - Bundles (`b`): All valid (item_a, item_b) pairs with a bundle bonus for OSLO_NEW_CARS as of 2026-05-07.
    - Incompatible pairs (`(i,j)`): All valid pairs of mutually incompatible items for OSLO_NEW_CARS as of 2026-05-07.
    - Requires pairs (`(i,k)`): All valid (item, prerequisite) pairs for OSLO_NEW_CARS as of 2026-05-07.
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity ordered of item/option `i`. Type: GRB.INTEGER. Must be zero if unauthorized.
    -   `z[i]` = Binary, 1 if item `i` is selected (i.e., q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `y[g]` = Binary, 1 if any item in category `g` is selected, 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = Binary, 1 if both items in bundle `b` are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit components: From `benefit` table, sum all signed `amount` (converted to USD cents using `fx` table) for each item, after as-of filtering.
    -   Item fees: From `item_fee` table, `activation_fee_cents` for each item, after as-of filtering.
    -   Category limits and activation fees: From `category` table, `minimum_quantity`, `maximum_quantity`, `activation_fee_cents` for each category, after as-of filtering.
    -   Item eligibility and lot/order bounds: From `item` table, `authorized`, `minimum_lot`, `maximum_order` for each item, after as-of filtering.
    -   Resource usage per unit: From `usage` table, `amount` and `unit` for each (item, resource), after as-of filtering and unit conversion (liter→ml, hour→minute, kwh→wh).
    -   Resource capacity: From `capacity_ledger` table, sum all signed `amount` for each resource, after as-of filtering and unit conversion.
    -   Bundle bonuses: From `bundle` table, `bonus_cents` for each (item_a, item_b), after as-of filtering.
    -   Incompatible pairs: From `incompatible` table, all (item_a, item_b) pairs, after as-of filtering.
    -   Requires dependencies: From `requires` table, all (item_ref, prerequisite_ref) pairs, after as-of filtering.
6.  **Formulate Objective:** Maximize total net benefit in USD cents, defined as:
    -   Sum over all items: (total per-unit benefit in USD cents) × q[i]
    -   Minus: sum of item_fee for each item with q[i] > 0 (deducted once per selected item)
    -   Minus: sum of category activation_fee_cents for each category with at least one selected item (deducted once per used category)
    -   Plus: sum of bundle bonus_cents for each bundle where both items are selected (awarded once per qualifying bundle)
7.  **Formulate Constraints:**
    -   **Item eligibility and lot/order bounds:** For each item `i`:
        - If `authorized[i] == 0`, enforce q[i] = 0.
        - If `authorized[i] > 0`, enforce: minimum_lot[i] × z[i] ≤ q[i] ≤ maximum_order[i] × z[i], and q[i] ≥ 0 integer.
        - z[i] = 1 iff q[i] > 0; z[i] = 0 iff q[i] = 0.
    -   **Category quantity limits:** For each category `g`, sum of q[i] over all items in `g` must satisfy: minimum_quantity[g] ≤ sum_{i in g} q[i] ≤ maximum_quantity[g].
        - y[g] = 1 iff any z[i] in category g is 1; y[g] = 0 otherwise.
    -   **Resource capacity constraints:** For each resource `r`, sum over all items: (resource usage per unit for i, in canonical units) × q[i] ≤ total available capacity for r (in same units).
    -   **Item fees:** For each item, item_fee is charged once if z[i] = 1 (i.e., q[i] > 0), not per unit.
    -   **Category activation fees:** For each category, activation_fee_cents is charged once if y[g] = 1 (i.e., any item in g is selected), not per unit.
    -   **Bundle bonuses:** For each bundle (item_a, item_b), w[b] = 1 iff both z[item_a] = 1 and z[item_b] = 1; w[b] = 0 otherwise. Award bonus_cents × w[b] once per bundle.
    -   **Incompatible pairs:** For each incompatible pair (i, j), enforce z[i] + z[j] ≤ 1 (cannot select both).
    -   **Requires dependencies:** For each (i, k), enforce: if z[i] = 1 then q[k] ≥ 1 (prerequisite must have positive quantity if dependent is selected).
[Abstract Model Plan END]
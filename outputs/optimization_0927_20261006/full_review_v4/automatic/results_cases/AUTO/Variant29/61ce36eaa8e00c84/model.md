[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized bread option to maximize net benefit (total unit benefit plus bundle bonuses, minus item and category activation fees), subject to storage, staff time, and energy resource limits, category-level minimum and maximum order quantities, item-level lot and order bounds, incompatibility and prerequisite requirements, and bundle bonus eligibility.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, logical constraints (incompatibility, prerequisites), and bundle bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items: All rows in the 'item' table with authorized=1.
    - Categories: All rows in the 'category' table.
    - Resources: All unique resources in the 'usage' and 'capacity_ledger' tables.
    - Incompatible Pairs: All rows in the 'incompatible' table.
    - Prerequisite Pairs: All rows in the 'requires' table.
    - Bundles: All rows in the 'bundle' table.
4.  **Define Decision Variables:**
    - `q[i]` = Order quantity of item i (authorized items only). Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]].
    - `y[i]` = 1 if item i is ordered (q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    - `z[c]` = 1 if any item in category c is ordered, 0 otherwise. Type: GRB.BINARY.
    - `b[bundle]` = 1 if both items in bundle are ordered (q[item_a] > 0 and q[item_b] > 0 and both authorized), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - unit_benefit_cents[i] (from 'item' table)
        - item_fee_cents[i] (from 'item' table)
        - activation_fee_cents[c] (from 'category' table)
        - bonus_cents[bundle] (from 'bundle' table)
    - Constraint coefficients:
        - minimum_lot[i], maximum_order[i] (from 'item' table)
        - minimum_quantity[c], maximum_quantity[c] (from 'category' table)
        - usage: amount[i,r], unit[i,r] (from 'usage' table)
        - capacity_ledger: sum of amount[r], unit[r] (from 'capacity_ledger' table)
    - Logical constraints:
        - incompatible pairs (from 'incompatible' table)
        - prerequisites (from 'requires' table)
        - authorized[i] (from 'item' table)
6.  **Formulate Objective:** Maximize total net benefit in cents:
    - Sum over items: unit_benefit_cents[i] * q[i]
    - Plus: sum over bundles: bonus_cents[bundle] * b[bundle]
    - Minus: sum over items: item_fee_cents[i] * y[i]
    - Minus: sum over categories: activation_fee_cents[c] * z[c]
7.  **Formulate Constraints:**
    - **Item selection and bounds:** For each authorized item i:
        - q[i] = 0 or minimum_lot[i] ≤ q[i] ≤ maximum_order[i]
        - y[i] = 1 if q[i] > 0, y[i] = 0 if q[i] = 0 (enforced via: q[i] ≥ minimum_lot[i] * y[i], q[i] ≤ maximum_order[i] * y[i])
    - **Category activation:** For each category c:
        - z[c] = 1 if any y[i]=1 for i in c; z[c] = 0 otherwise (enforced via: y[i] ≤ z[c] for all i in c; z[c] ≤ sum_{i in c} y[i])
    - **Category quantity bounds:** For each category c:
        - minimum_quantity[c] ≤ sum_{i in c} q[i] ≤ maximum_quantity[c]
    - **Resource constraints:** For each resource r:
        - sum_{i} (usage amount[i,r] * q[i], converted to the unit of capacity_ledger[r]) ≤ sum of capacity_ledger amounts for r (converted to same unit)
    - **Incompatibility:** For each incompatible pair (i, j):
        - y[i] + y[j] ≤ 1
    - **Prerequisites:** For each (i requires j):
        - y[i] ≤ y[j]
    - **Bundle bonuses:** For each bundle (item_a, item_b):
        - b[bundle] ≤ y[item_a], b[bundle] ≤ y[item_b], b[bundle] ≥ y[item_a] + y[item_b] - 1
        - b[bundle] = 0 if either item is unauthorized
[Abstract Model Plan END]
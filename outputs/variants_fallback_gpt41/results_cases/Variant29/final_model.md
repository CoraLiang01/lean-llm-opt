[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized bread option to maximize net benefit (in USD cents), considering per-unit benefits, fixed item and category activation fees, resource constraints (storage, staff time, energy), category-level minimum and maximum aggregate quantities, incompatibility and prerequisite relationships, and bundle bonuses. Only authorized options may be ordered, and all constraints and bonuses must be respected.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee), resource allocation, logical (incompatibility/prerequisite), and bundle bonus features.
3.  **Define Index Sets:** The primary indices are:
    - Items (authorized bread options, from 'item' table where authorized=1)
    - Categories (from 'category' table)
    - Resources (from 'usage' and 'capacity_ledger' tables)
    - Incompatible pairs (from 'incompatible' table)
    - Prerequisite pairs (from 'requires' table)
    - Bundles (from 'bundle' table)
4.  **Define Decision Variables:**
    - `q[i]` = Integer quantity ordered of item i (authorized items only). Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]]
    - `y[i]` = 1 if item i is ordered (q[i] > 0), 0 otherwise. Type: GRB.BINARY
    - `z[c]` = 1 if any item in category c is ordered, 0 otherwise. Type: GRB.BINARY
    - `b[p]` = 1 if both items in bundle pair p are ordered (i.e., both y[i_a] and y[i_b] are 1), 0 otherwise. Type: GRB.BINARY
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - Per-unit benefit: 'unit_benefit_cents' (from 'item' table)
        - Fixed item fee: 'item_fee_cents' (from 'item' table)
        - Category activation fee: 'activation_fee_cents' (from 'category' table)
        - Bundle bonus: 'bonus_cents' (from 'bundle' table)
    - Constraint coefficients:
        - Per-unit resource usage: 'amount' (from 'usage' table), with 'resource' and 'unit'
        - Resource capacity: sum of 'amount' (from 'capacity_ledger' table), per resource, with 'unit'
    - Constraint RHS:
        - Category minimum/maximum: 'minimum_quantity', 'maximum_quantity' (from 'category' table)
        - Item minimum/maximum: 'minimum_lot', 'maximum_order' (from 'item' table)
    - Logical relationships:
        - Incompatibility: ('item_a', 'item_b') pairs (from 'incompatible' table)
        - Prerequisites: ('item_ref', 'prerequisite_ref') pairs (from 'requires' table)
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over items: (unit_benefit_cents[i] * q[i]) − (item_fee_cents[i] * y[i])
    - Minus sum over categories: (activation_fee_cents[c] * z[c])
    - Plus sum over bundles: (bonus_cents[p] * b[p])
7.  **Formulate Constraints:**
    - **Item selection and quantity bounds:** For each authorized item i:
        - q[i] = 0 or minimum_lot[i] ≤ q[i] ≤ maximum_order[i]
        - y[i] = 1 if q[i] > 0, y[i] = 0 if q[i] = 0 (enforced via: q[i] ≥ minimum_lot[i] * y[i], q[i] ≤ maximum_order[i] * y[i])
    - **Category activation:** For each category c:
        - z[c] = 1 if any y[i] = 1 for i in c; z[c] = 0 otherwise (enforced via: y[i] ≤ z[c] for all i in c; z[c] ≤ sum_{i in c} y[i])
    - **Category aggregate quantity bounds:** For each category c:
        - minimum_quantity[c] ≤ sum_{i in c} q[i] ≤ maximum_quantity[c]
    - **Resource constraints:** For each resource r:
        - sum over items i: (per-unit usage of r by i, converted to resource's base unit) * q[i] ≤ total available capacity of r (sum of 'amount' in 'capacity_ledger' for r, converted to base unit)
    - **Incompatibility:** For each incompatible pair (i, j):
        - y[i] + y[j] ≤ 1
    - **Prerequisites:** For each (i, prereq) in 'requires':
        - y[i] ≤ y[prereq]
    - **Bundle bonuses:** For each bundle pair (i_a, i_b):
        - b[p] ≤ y[i_a], b[p] ≤ y[i_b], b[p] ≥ y[i_a] + y[i_b] − 1
        - Only include bundle if both items are authorized (otherwise, b[p] = 0)
    - **Variable domains:** q[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer), y[i], z[c], b[p] ∈ {0,1}
[Abstract Model Plan END]
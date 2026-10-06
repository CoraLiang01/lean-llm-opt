[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized bread option to maximize net benefit (in USD cents), considering per-unit benefits, fixed item and category fees, bundle bonuses, resource limits (storage, staff time, energy), category quantity bounds, incompatibilities, and prerequisite requirements. Only authorized options may be ordered, and all constraints and bonuses must be respected.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation) costs, resource constraints, logical (incompatibility/prerequisite) constraints, and bundle bonuses.
3.  **Define Index Sets:** The primary indices are:
    - Items (bread options): All rows in batch_03/export_03.csv with authorized = 1.
    - Categories: All rows in batch_05/export_05.csv.
    - Resources: All unique resources in batch_04/export_04.csv and batch_09/export_09.csv.
    - Bundles: All rows in batch_08/export_08.csv.
    - Incompatible pairs: All rows in batch_06/export_06.csv.
    - Prerequisite pairs: All rows in batch_01/export_01.csv.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity ordered of item i (i in authorized items). Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]].
    -   `y[i]` = 1 if item i is ordered (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are ordered (positive quantities), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        - Per-unit benefit: 'unit_benefit_cents' from batch_03/export_03.csv.
        - Fixed item fee: 'item_fee_cents' from batch_03/export_03.csv.
        - Category activation fee: 'activation_fee_cents' from batch_05/export_05.csv.
        - Bundle bonus: 'bonus_cents' from batch_08/export_08.csv.
    -   Constraint coefficients:
        - Per-unit resource usage: 'amount' and 'unit' from batch_04/export_04.csv (convert units to match capacity_ledger).
        - Resource capacities: sum of 'amount' in batch_09/export_09.csv for each resource (convert units as needed).
        - Category minimum/maximum: 'minimum_quantity', 'maximum_quantity' from batch_05/export_05.csv.
        - Item minimum/maximum: 'minimum_lot', 'maximum_order' from batch_03/export_03.csv.
        - Incompatibilities: pairs from batch_06/export_06.csv.
        - Prerequisites: pairs from batch_01/export_01.csv.
    -   Only items with authorized = 1 (from batch_03/export_03.csv) are eligible for ordering or for triggering bonuses/prerequisites.
6.  **Formulate Objective:** Maximize total net benefit in cents:
    - Sum over items: (unit_benefit_cents[i] * x[i]) 
    - Minus sum over items: (item_fee_cents[i] * y[i]) [fixed fee if any ordered]
    - Minus sum over categories: (activation_fee_cents[c] * z[c]) [fixed fee if any item in category ordered]
    - Plus sum over bundles: (bonus_cents[bundle] * b[bundle]) [bonus if both items in bundle are ordered, both authorized]
7.  **Formulate Constraints:**
    -   **Item Order Bounds:** For each authorized item i:
        - x[i] = 0, or minimum_lot[i] ≤ x[i] ≤ maximum_order[i].
        - Enforced via: x[i] ≥ y[i] * minimum_lot[i], x[i] ≤ y[i] * maximum_order[i], y[i] ∈ {0,1}.
    -   **Category Quantity Bounds:** For each category c:
        - sum_{i in c} x[i] ≥ minimum_quantity[c]
        - sum_{i in c} x[i] ≤ maximum_quantity[c]
        - z[c] ≥ y[i] for any i in c (z[c] = 1 if any item in c is ordered)
    -   **Resource Constraints:** For each resource r:
        - sum_{i} (resource_usage[i,r] * x[i]) ≤ total_capacity[r]
        - All units must be converted to match (e.g., kwh→wh, liter→ml, hour→minute).
    -   **Incompatibility Constraints:** For each incompatible pair (i, j):
        - y[i] + y[j] ≤ 1 (cannot order both)
    -   **Prerequisite Constraints:** For each (i requires j):
        - y[i] ≤ y[j] (if i is ordered, j must also be ordered)
    -   **Bundle Bonus Constraints:** For each bundle (i, j):
        - b[bundle] ≤ y[i], b[bundle] ≤ y[j], b[bundle] ≥ y[i] + y[j] - 1
        - b[bundle] = 1 only if both i and j are ordered (x[i] > 0, x[j] > 0), and both are authorized
    -   **Authorization Constraint:** Only items with authorized = 1 may have x[i] > 0 or y[i] = 1; all others must have x[i] = 0, y[i] = 0.
[Abstract Model Plan END]
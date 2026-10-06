[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer number of cases to order for each authorized product for the CENTRAL_FRESH supermarket, maximizing net return (in USD cents) from the order. The model must account for per-unit benefits, fixed item and category fees, resource usage and capacity, category quantity bounds, incompatibilities, prerequisites, and bundle bonuses, using only the supplied item rows.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (from export_06.csv, filtered to authorized=1)
    - Categories (from export_03.csv)
    - Resources (from export_09.csv and export_02.csv)
    - Bundles (from export_01.csv)
    - Incompatible pairs (from export_05.csv)
    - Prerequisite pairs (from export_08.csv)
4.  **Define Decision Variables:**
    -   `x[i]` = Number of cases of item `i` to order. Type: GRB.INTEGER (nonnegative, zero or at least minimum_lot if positive).
    -   `y[i]` = 1 if any quantity of item `i` is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category `c` is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle `bundle` are ordered (positive quantity), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        - Per-unit benefit: `unit_benefit_cents` (export_06.csv)
        - Fixed item fee: `item_fee_cents` (export_06.csv)
        - Category activation fee: `activation_fee_cents` (export_03.csv)
        - Bundle bonus: `bonus_cents` (export_01.csv)
    -   Constraint coefficients:
        - Resource usage per unit: `amount` and `unit` (export_09.csv)
        - Resource capacity: sum of `amount` in capacity_ledger (export_02.csv), with unit conversion as needed
        - Category membership: `category` (export_06.csv)
        - Category quantity bounds: `minimum_quantity`, `maximum_quantity` (export_03.csv)
        - Incompatibilities: pairs from export_05.csv
        - Prerequisites: pairs from export_08.csv
        - Authorization: `authorized` (export_06.csv, only authorized=1 items are eligible)
        - Minimum/maximum order: `minimum_lot`, `maximum_order` (export_06.csv)
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over items: (unit_benefit_cents * x[i]) 
    - Minus sum over items: (item_fee_cents * y[i]) [fixed fee if any of item i is ordered]
    - Minus sum over categories: (activation_fee_cents * z[c]) [fixed fee if any item in category c is ordered]
    - Plus sum over bundles: (bonus_cents * b[bundle]) [bonus if both items in bundle are ordered]
7.  **Formulate Constraints:**
    -   **Item selection and bounds:**
        - For each item i: x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i] (enforced via y[i])
        - For each item i: x[i] ≥ minimum_lot[i] * y[i]; x[i] ≤ maximum_order[i] * y[i]; x[i] ≥ 0; integer
        - For each item i: y[i] ∈ {0,1}
        - Only items with authorized=1 may have x[i] > 0; others must have x[i]=0, y[i]=0
    -   **Category activation:**
        - For each category c: z[c] = 1 if any item in c is ordered (i.e., y[i]=1 for any i in c), else 0
        - For each category c: sum over i in c of x[i] ≥ minimum_quantity[c] * z[c]
        - For each category c: sum over i in c of x[i] ≤ maximum_quantity[c] * z[c]
        - For each category c: z[c] ∈ {0,1}
    -   **Resource capacity:**
        - For each resource r: sum over items i of (resource usage per unit of i for r, converted to base units) * x[i] ≤ total available capacity for r (sum of opening and reservation in capacity_ledger, with sign and unit conversion)
    -   **Incompatibility:**
        - For each incompatible pair (i, j): y[i] + y[j] ≤ 1
    -   **Prerequisite:**
        - For each (i, prereq): y[i] ≤ y[prereq] (if i is ordered, so must be its prerequisite)
    -   **Bundle bonuses:**
        - For each bundle (a, b): b[bundle] ≤ y[a], b[bundle] ≤ y[b], b[bundle] ≥ y[a] + y[b] - 1 (b[bundle]=1 iff both y[a]=1 and y[b]=1)
        - Only bundles where both items are authorized can be triggered (if either is unauthorized, b[bundle]=0)
    -   **Variable domains:**
        - All x[i] are integer, y[i], z[c], b[bundle] are binary
[Abstract Model Plan END]
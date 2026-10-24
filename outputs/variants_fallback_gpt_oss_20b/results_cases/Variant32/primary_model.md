**Mathematical Model**

Let  

* \(I\) – set of items (vehicle options)  
* \(C\) – set of categories  
* \(R\) – set of resources (space, labor, power)  
* \(B\) – set of bundles  
* \(K\) – set of incompatible pairs  
* \(L\) – set of requires pairs  

Parameters (all in USD cents unless otherwise noted)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(\text{benefit}_{i}\) | Net benefit per unit of item \(i\) (sum of all benefit components after currency conversion) | `benefit`, `fx` |
| \(\text{item\_fee}_{i}\) | Item‑level activation fee | `item_fee` |
| \(\text{cat\_fee}_{c}\) | Category activation fee | `category` |
| \(\text{min\_lot}_{i}\) | Minimum lot size for item \(i\) | `item` |
| \(\text{max\_order}_{i}\) | Maximum order for item \(i\) | `item` |
| \(\text{auth}_{i}\) | Authorization flag (0 = unauthorized) | `item` |
| \(\text{cat}_{i}\) | Category of item \(i\) | `item` |
| \(\text{min\_qty}_{c}\) | Minimum total quantity for category \(c\) | `category` |
| \(\text{max\_qty}_{c}\) | Maximum total quantity for category \(c\) | `category` |
| \(\text{usage}_{r,i}\) | Resource consumption per unit of item \(i\) (already in common units) | `usage` |
| \(\text{cap}_{r}\) | Capacity limit for resource \(r\) | `capacity_ledger` |
| \(\text{bonus}_{b}\) | Bundle bonus for bundle \(b\) | `bundle` |
| \(\text{bundle\_items}_{b}=(a,b)\) | Items in bundle \(b\) | `bundle` |
| \(\text{incomp}_{k}=(a,b)\) | Incompatible pair \(k\) | `incompatible` |
| \(\text{requires}_{l}=(a,b)\) | Requires pair \(l\) (item \(a\) requires item \(b\)) | `requires` |
| \(\text{fx\_fac}_{c}\) | Conversion factor \(=\frac{\text{usd\_cents\_numerator}}{\text{denominator}}\) for currency \(c\) | `fx` |

Decision variables

| Symbol | Type | Bounds | Meaning |
|--------|------|--------|---------|
| \(q_i\) | integer | \(\ge 0\) | Quantity of item \(i\) ordered |
| \(y_i\) | binary | \(\{0,1\}\) | 1 if \(q_i>0\) |
| \(z_c\) | binary | \(\{0,1\}\) | 1 if any item in category \(c\) is ordered |
| \(w_b\) | binary | \(\{0,1\}\) | 1 if both items in bundle \(b\) are ordered |
| \(k_i\) | integer | \(\ge 0\) | Lot multiplier for item \(i\) (so \(q_i=\text{min\_lot}_i\,k_i\)) |

**Constraints**

1. **Authorization**  
   \[
   y_i \le \frac{\text{auth}_i}{1}\quad \forall i
   \]

2. **Lot multiple**  
   \[
   q_i = \text{min\_lot}_i\,k_i \quad \forall i
   \]
   \[
   k_i \le \left\lfloor\frac{\text{max\_order}_i}{\text{min\_lot}_i}\right\rfloor\,y_i \quad \forall i
   \]
   \[
   k_i \ge y_i \quad \forall i
   \]

3. **Category quantity limits**  
   \[
   Q_c = \sum_{i:\text{cat}_i=c} q_i \quad \forall c
   \]
   \[
   Q_c \ge \text{min\_qty}_c\,z_c \quad \forall c
   \]
   \[
   Q_c \le \text{max\_qty}_c\,z_c \quad \forall c
   \]
   \[
   z_c \le \sum_{i:\text{cat}_i=c} y_i \quad \forall c
   \]

4. **Incompatibilities**  
   \[
   y_{a} + y_{b} \le 1 \quad \forall (a,b)\in \text{incomp}
   \]

5. **Requires**  
   \[
   q_{b} \ge y_{a} \quad \forall (a,b)\in \text{requires}
   \]

6. **Bundle activation**  
   \[
   w_b \le y_{a},\; w_b \le y_{b} \quad \forall (a,b)\in \text{bundle\_items}_b
   \]
   \[
   w_b \ge y_{a}+y_{b}-1 \quad \forall b
   \]

7. **Resource capacities**  
   \[
   \sum_{i} \text{usage}_{r,i}\,q_i \le \text{cap}_r \quad \forall r
   \]

8. **Variable domains**  
   \[
   q_i \in \mathbb{Z}_{\ge0},\; y_i,z_c,w_b,k_i \in \{0,1\}\quad \forall i,c,b
   \]

**Objective**

\[
\max \;\; \sum_{i}\text{benefit}_{i}\,q_i
      \;-\; \sum_{i}\text{item\_fee}_{i}\,y_i
      \;-\; \sum_{c}\text{cat\_fee}_{c}\,z_c
      \;+\; \sum_{b}\text{bonus}_{b}\,w_b
\]

All terms are in USD cents; the objective is the maximum net benefit.

---

**Data Mapping Rules**

1. **Record Selection**  
   For every table (`benefit`, `bundle`, `capacity_ledger`, `category`, `fx`, `identity`, `incompatible`, `item`, `item_fee`, `market`, `requires`, `usage`):  
   * Keep only rows with `dealership_id = 'OSLO_NEW_CARS'` and `effective_date ≤ 2026‑05‑07`.  
   * For each `(dealership_id, table, record_id)` keep the row with the highest integer `revision`; discard the row if that revision’s `action = 'DELETE'`.  
   * Deduplicate identical retransmissions (same `record_id` and `revision`).

2. **Benefit per Unit**  
   * For each item \(i\), sum all `amount` values from the `benefit` table where `item_ref = i`.  
   * Convert each `amount` to USD cents:  
     \[
     \text{amount}_{\text{USDcents}} = \text{amount}\times \frac{\text{usd\_cents\_numerator}}{\text{denominator}}
     \]  
     using the matching `fx` record for the `currency`.  
   * \(\text{benefit}_{i}\) is the signed sum of these converted amounts.

3. **Item‑level Fees**  
   * `item_fee` table provides `activation_fee_cents` for each `item_ref`; map to \(\text{item\_fee}_{i}\).

4. **Category Fees and Limits**  
   * `category` table provides `activation_fee_cents`, `minimum_quantity`, `maximum_quantity` for each `category`; map to \(\text{cat\_fee}_{c}\), \(\text{min\_qty}_{c}\), \(\text{max\_qty}_{c}\).

5. **Lot, Max Order, Authorization, Category**  
   * `item` table provides `minimum_lot`, `maximum_order`, `authorized`, `category`; map to \(\text{min\_lot}_{i}\), \(\text{max\_order}_{i}\), \(\text{auth}_{i}\), \(\text{cat}_{i}\).

6. **Resource Usage**  
   * `usage` table provides `amount` per `resource` and `item_ref`; amounts are already converted to common units (ml, minute, wh) per `unit_preparation`.  
   * Map to \(\text{usage}_{r,i}\).

7. **Capacity Limits**  
   * `capacity_ledger` table provides `amount` per `resource`; map to \(\text{cap}_{r}\).

8. **Bundles**  
   * `bundle` table provides `bonus_cents` and `item_a`, `item_b`; map to \(\text{bonus}_{b}\) and \(\text{bundle\_items}_{b}\).

9. **Incompatibilities**  
   * `incompatible` table provides pairs `item_a`, `item_b`; map to \(\text{incomp}_{k}\).

10. **Requires**  
    * `requires` table provides `item_ref` and `prerequisite_ref`; map to \(\text{requires}_{l}\).

All mappings use the exact identifiers from the source tables; no additional records are introduced. The model uses only the filtered OSLO_NEW_CARS data as of 2026‑05‑07.
**Mathematical Model**

Let  

* \(I\) – set of items (item_ref from the *item* table)  
* \(C\) – set of categories (category from the *category* table)  
* \(B\) – set of bundles (rows of the *bundle* table)  
* \(R\) – set of resources (resource from the *usage* and *capacity_ledger* tables)  
* \(L\) – set of incompatible pairs (rows of the *incompatible* table)  
* \(P\) – set of prerequisite pairs (rows of the *requires* table)

**Parameters**

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(u_i\) | unit benefit (cents) of item \(i\) | *item.unit_benefit_cents* |
| \(f_i\) | item fee (cents) if \(i\) is ordered | *item.item_fee_cents* |
| \(m_i\) | minimum lot size of item \(i\) | *item.minimum_lot* |
| \(M_i\) | maximum order of item \(i\) | *item.maximum_order* |
| \(cat_i\) | category of item \(i\) | *item.category* |
| \(a_c\) | activation fee (cents) of category \(c\) | *category.activation_fee_cents* |
| \(b_b\) | bonus (cents) of bundle \(b\) | *bundle.bonus_cents* |
| \(a_b\) | first item of bundle \(b\) | *bundle.item_a* |
| \(b_b\) | second item of bundle \(b\) | *bundle.item_b* |
| \(w_{i,r}\) | usage of resource \(r\) per unit of item \(i\) (already converted) | *usage.amount* (after unit conversion) |
| \(cap_r\) | total available capacity of resource \(r\) | sum of *capacity_ledger.amount* for resource \(r\) |

**Decision Variables**

| Symbol | Domain | Meaning |
|--------|--------|---------|
| \(q_i\) | \(\mathbb{Z}_{\ge 0}\) | quantity of item \(i\) (units) |
| \(y_i\) | \(\{0,1\}\) | 1 if item \(i\) is ordered (quantity \(>0\)) |
| \(z_c\) | \(\{0,1\}\) | 1 if any item of category \(c\) is ordered |
| \(w_b\) | \(\{0,1\}\) | 1 if both items of bundle \(b\) are ordered |

**Objective**

\[
\max \;\; \sum_{i\in I} u_i\,q_i \;-\; \sum_{i\in I} f_i\,y_i \;-\; \sum_{c\in C} a_c\,z_c \;+\; \sum_{b\in B} b_b\,w_b
\]

**Constraints**

1. **Quantity bounds (lot size & max order)**  
   \[
   m_i\,y_i \;\le\; q_i \;\le\; M_i\,y_i \qquad \forall i\in I
   \]

2. **Category activation**  
   \[
   z_c \;\le\; \sum_{i:cat_i=c} y_i \qquad \forall c\in C
   \]
   \[
   y_i \;\le\; z_{cat_i} \qquad \forall i\in I
   \]

3. **Incompatibility**  
   \[
   y_{a} + y_{b} \;\le\; 1 \qquad \forall (a,b)\in L
   \]

4. **Prerequisite**  
   \[
   y_{i} \;\le\; y_{\text{prereq}(i)} \qquad \forall (i,\text{prereq}(i))\in P
   \]

5. **Bundle bonus**  
   \[
   w_b \;\le\; y_{a_b} \qquad \forall b\in B
   \]
   \[
   w_b \;\le\; y_{b_b} \qquad \forall b\in B
   \]
   \[
   w_b \;\ge\; y_{a_b} + y_{b_b} - 1 \qquad \forall b\in B
   \]

6. **Resource capacities**  
   \[
   \sum_{i\in I} w_{i,r}\,q_i \;\le\; cap_r \qquad \forall r\in R
   \]

All variables are integer or binary as indicated.

---

### Data Mapping Rules

| Index / Parameter | Source Table | Source Column | Notes |
|-------------------|--------------|---------------|-------|
| \(I\) | *item* | item_ref | All rows with authorized=1 (filter applied) |
| \(C\) | *category* | category | One row per category |
| \(B\) | *bundle* | table | All 8 bundle rows |
| \(R\) | *usage* + *capacity_ledger* | resource | Distinct resource names |
| \(L\) | *incompatible* | (item_a, item_b) | All 10 pairs |
| \(P\) | *requires* | (item_ref, prerequisite_ref) | All 12 pairs |
| \(u_i\) | *item* | unit_benefit_cents | |
| \(f_i\) | *item* | item_fee_cents | |
| \(m_i\) | *item* | minimum_lot | |
| \(M_i\) | *item* | maximum_order | |
| \(cat_i\) | *item* | category | |
| \(a_c\) | *category* | activation_fee_cents | |
| \(b_b\) | *bundle* | bonus_cents | |
| \(a_b\) | *bundle* | item_a | |
| \(b_b\) | *bundle* | item_b | |
| \(w_{i,r}\) | *usage* | amount (converted) | After unit conversion (ml, wh, minute) |
| \(cap_r\) | *capacity_ledger* | amount (converted) | Sum over all ledger entries for resource \(r\) |

All numeric values are taken directly from the source tables; no additional aggregation or conversion is performed beyond the unit conversions already applied in the observation. The model uses the full set of returned records, respecting the query‑defined filters (e.g., authorized=1).
## Mathematical Model

**Indices**  
\(I\) – set of items (authorized only)  
\(C\) – set of categories  
\(R\) – set of resources  
\(B\) – set of bundles  
\(K\) – set of incompatible pairs  
\(P\) – set of prerequisite pairs  
\(L\) – set of capacity‑ledger entries  

**Parameters**  

| Symbol | Meaning | Source table / column |
|--------|---------|-----------------------|
| \(unit\_benefit\_cents[i]\) | benefit per unit of item \(i\) | `item.unit_benefit_cents` |
| \(item\_fee\_cents[i]\) | fixed fee if item \(i\) is ordered | `item.item_fee_cents` |
| \(min\_lot[i]\) | minimum lot size for item \(i\) | `item.minimum_lot` |
| \(max\_order[i]\) | maximum order size for item \(i\) | `item.maximum_order` |
| \(category[i]\) | category of item \(i\) | `item.category` |
| \(activation\_fee\_cents[c]\) | fee for activating category \(c\) | `category.activation_fee_cents` |
| \(min\_quantity[c]\) | minimum total quantity for category \(c\) | `category.minimum_quantity` |
| \(max\_quantity[c]\) | maximum total quantity for category \(c\) | `category.maximum_quantity` |
| \(bonus\_cents[b]\) | bonus for bundle \(b\) | `bundle.bonus_cents` |
| \(item\_a[b], item\_b[b]\) | items in bundle \(b\) | `bundle.item_a`, `bundle.item_b` |
| \(incompatible\_a[k], incompatible\_b[k]\) | items in incompatible pair \(k\) | `incompatible.item_a`, `incompatible.item_b` |
| \(prerequisite[i]\) | prerequisite of item \(i\) | `requires.prerequisite_ref` |
| \(usage[i,r]\) | per‑unit usage of resource \(r\) by item \(i\) | `usage.amount` (converted to resource units) |
| \(capacity[r]\) | total available amount of resource \(r\) | sum of `capacity_ledger.amount` for resource \(r\) |

**Decision Variables**

| Symbol | Domain | Meaning |
|--------|--------|---------|
| \(q_i\) | \(\mathbb{Z}_{+}\) | quantity (100 kg units) of item \(i\) |
| \(y_i\) | \(\{0,1\}\) | 1 if item \(i\) is ordered, 0 otherwise |
| \(z_c\) | \(\{0,1\}\) | 1 if any item in category \(c\) is ordered |
| \(b_b\) | \(\{0,1\}\) | 1 if both items of bundle \(b\) are ordered |

---

### Objective

\[
\max \;\sum_{i\in I} unit\_benefit\_cents[i]\;q_i
\;-\;\sum_{i\in I} item\_fee\_cents[i]\;y_i
\;-\;\sum_{c\in C} activation\_fee\_cents[c]\;z_c
\;+\;\sum_{b\in B} bonus\_cents[b]\;b_b
\]

---

### Constraints

1. **Quantity bounds (lot size)**  
   \[
   min\_lot[i]\;y_i \;\le\; q_i \;\le\; max\_order[i]\;y_i
   \quad \forall i\in I
   \]

2. **Category activation**  
   \[
   z_c \;\ge\; y_i \quad \forall i\in I:\;category[i]=c
   \]
   \[
   z_c \;\le\; \sum_{i\in I:\;category[i]=c} y_i
   \quad \forall c\in C
   \]

3. **Category quantity limits**  
   \[
   \sum_{i\in I:\;category[i]=c} q_i
   \;\ge\; min\_quantity[c]\;z_c
   \quad \forall c\in C
   \]
   \[
   \sum_{i\in I:\;category[i]=c} q_i
   \;\le\; max\_quantity[c]\;z_c
   \quad \forall c\in C
   \]

4. **Incompatibility**  
   \[
   y_{incompatible\_a[k]} + y_{incompatible\_b[k]} \;\le\; 1
   \quad \forall k\in K
   \]

5. **Prerequisite**  
   \[
   y_i \;\le\; y_{prerequisite[i]}
   \quad \forall i\in I \text{ with a prerequisite}
   \]

6. **Bundle bonus**  
   \[
   b_b \;\le\; y_{item\_a[b]}, \qquad
   b_b \;\le\; y_{item\_b[b]}
   \quad \forall b\in B
   \]
   \[
   b_b \;\ge\; y_{item\_a[b]} + y_{item\_b[b]} - 1
   \quad \forall b\in B
   \]

7. **Resource capacity**  
   \[
   \sum_{i\in I}\sum_{r\in R} usage[i,r]\;q_i
   \;\le\; capacity[r]
   \quad \forall r\in R
   \]

---

## Data Mapping Rules

| Index / Parameter | Source table | Source column | Notes |
|-------------------|--------------|---------------|-------|
| \(I\) | `item` | `item_ref` | Only rows with `authorized = 1` |
| \(C\) | `category` | `category` | All categories present |
| \(R\) | `capacity_ledger` | `resource` | All resources with ledger entries |
| \(B\) | `bundle` | `item_a`, `item_b`, `bonus_cents` | All bundles |
| \(K\) | `incompatible` | `item_a`, `item_b` | All incompatible pairs |
| \(P\) | `requires` | `item_ref`, `prerequisite_ref` | All prerequisite pairs |
| \(L\) | `capacity_ledger` | `amount` | Sum per resource |
| \(unit\_benefit\_cents[i]\) | `item` | `unit_benefit_cents` | |
| \(item\_fee\_cents[i]\) | `item` | `item_fee_cents` | |
| \(min\_lot[i]\) | `item` | `minimum_lot` | |
| \(max\_order[i]\) | `item` | `maximum_order` | |
| \(category[i]\) | `item` | `category` | |
| \(activation\_fee\_cents[c]\) | `category` | `activation_fee_cents` | |
| \(min\_quantity[c]\) | `category` | `minimum_quantity` | |
| \(max\_quantity[c]\) | `category` | `maximum_quantity` | |
| \(bonus\_cents[b]\) | `bundle` | `bonus_cents` | |
| \(item\_a[b], item\_b[b]\) | `bundle` | `item_a`, `item_b` | |
| \(incompatible\_a[k], incompatible\_b[k]\) | `incompatible` | `item_a`, `item_b` | |
| \(prerequisite[i]\) | `requires` | `prerequisite_ref` | |
| \(usage[i,r]\) | `usage` | `amount` (converted) | Converted to resource units (ml, minute, wh) |
| \(capacity[r]\) | `capacity_ledger` | `amount` (converted) | Sum of all ledger entries for resource \(r\) |

All numeric values are taken directly from the source tables; no additional scaling or aggregation is performed beyond the unit conversions already applied in the literal‑query‑unit‑view. The model uses only the entities present in the returned records and respects every restriction stated in the query.
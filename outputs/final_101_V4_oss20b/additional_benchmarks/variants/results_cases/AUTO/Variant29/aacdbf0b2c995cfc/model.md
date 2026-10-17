## Mathematical Model

**Indices**

- \(I\) – set of item identifiers \(i\) with `authorized = 1` (from *item* table).  
- \(C\) – set of category identifiers \(c\) (from *category* table).  
- \(R\) – set of resource identifiers \(r\) that appear in *usage* or *capacity_ledger*.  
- \(B\) – set of bundle identifiers \(b\) (from *bundle* table).  

**Parameters** (all taken directly from the source tables)

| Parameter | Source table & column | Meaning |
|-----------|-----------------------|---------|
| \(u_i\) | *item*, `unit_benefit_cents` | Benefit per 100 kg of item \(i\) |
| \(f_i\) | *item*, `item_fee_cents` | Fixed fee if any positive quantity of \(i\) is produced |
| \(\ell_i\) | *item*, `minimum_lot` | Minimum lot size (integer) |
| \(M_i\) | *item*, `maximum_order` | Maximum order size (integer) |
| \(cat_i\) | *item*, `category` | Category of item \(i\) |
| \(a_c\) | *category*, `activation_fee_cents` | Activation fee for category \(c\) |
| \(q^{\min}_c\) | *category*, `minimum_quantity` | Minimum total quantity for category \(c\) |
| \(q^{\max}_c\) | *category*, `maximum_quantity` | Maximum total quantity for category \(c\) |
| \(w_{i,r}\) | *usage*, `amount` (converted to common unit) | Consumption of resource \(r\) per unit of item \(i\) |
| \(cap_r\) | *capacity_ledger*, `amount` (converted) | Available capacity of resource \(r\) |
| \(req_{ij}\) | *requires*, binary indicator (1 if record exists) | Item \(i\) requires item \(j\) |
| \(inc_{ij}\) | *incompatible*, binary indicator (1 if record exists) | Items \(i\) and \(j\) cannot both be selected |
| \(b_b\) | *bundle*, `bonus_cents` | Bonus for bundle \(b\) |
| \(a_b\), \(b_b\) | *bundle*, `item_a`, `item_b` | Items that form bundle \(b\) |

All monetary parameters are in **cents**.  
Resource units are already converted to the common units:  
labor – minutes, power – watt‑hours, space – millilitres (conversion factors 1000 ml/l, 60 min/h, 1000 wh/kwh applied).

**Decision Variables**

| Variable | Domain | Meaning |
|----------|--------|---------|
| \(q_i\) | \(\mathbb{Z}_{\ge 0}\) | Quantity (in 100 kg units) of item \(i\) |
| \(y_i\) | \(\{0,1\}\) | 1 if any positive quantity of \(i\) is produced |
| \(z_c\) | \(\{0,1\}\) | 1 if any item in category \(c\) is produced |
| \(w_b\) | \(\{0,1\}\) | 1 if both items of bundle \(b\) are produced |

**Constraints**

1. **Lot size limits**  
   \[
   \ell_i\,y_i \;\le\; q_i \;\le\; M_i\,y_i \qquad \forall i\in I
   \]

2. **Resource capacities**  
   \[
   \sum_{i\in I} w_{i,r}\,q_i \;\le\; cap_r \qquad \forall r\in R
   \]

3. **Category quantity bounds**  
   \[
   q^{\min}_c\,z_c \;\le\; \sum_{i\in I,\;cat_i=c} q_i \;\le\; q^{\max}_c\,z_c
   \qquad \forall c\in C
   \]

4. **Requires relations**  
   \[
   y_i \;\le\; y_j \qquad \forall (i,j)\text{ with }req_{ij}=1
   \]

5. **Incompatible pairs**  
   \[
   y_i + y_j \;\le\; 1 \qquad \forall (i,j)\text{ with }inc_{ij}=1
   \]

6. **Bundle activation** (linearization of \(w_b = y_{a_b}\land y_{b_b}\))  
   \[
   w_b \;\le\; y_{a_b}, \quad
   w_b \;\le\; y_{b_b}, \quad
   w_b \;\ge\; y_{a_b}+y_{b_b}-1
   \qquad \forall b\in B
   \]

7. **Activation of categories**  
   \[
   z_c \;\ge\; y_i \qquad \forall i\in I,\;cat_i=c
   \]
   (ensures \(z_c=1\) if any item in \(c\) is selected).

**Objective**

\[
\max \;\;
\sum_{i\in I} u_i\,q_i
\;-\;
\sum_{i\in I} f_i\,y_i
\;-\;
\sum_{c\in C} a_c\,z_c
\;+\;
\sum_{b\in B} b_b\,w_b
\]

All terms are in **cents**; the model maximises net profit.

---

## Data‑Mapping Rules

| Source | Table ID | Relevant Columns | Parameter Mapping |
|--------|----------|------------------|-------------------|
| *item* | `file_2_view_0` | `item_ref`, `category`, `authorized`, `minimum_lot`, `maximum_order`, `unit_benefit_cents`, `item_fee_cents` | \(i\), \(cat_i\), \(\ell_i\), \(M_i\), \(u_i\), \(f_i\) |
| *category* | `file_4_view_0` | `category`, `activation_fee_cents`, `minimum_quantity`, `maximum_quantity` | \(c\), \(a_c\), \(q^{\min}_c\), \(q^{\max}_c\) |
| *usage* | `file_3_view_0` | `item_ref`, `resource`, `amount` (converted) | \(w_{i,r}\) |
| *capacity_ledger* | `file_8_view_0` | `resource`, `amount` (converted) | \(cap_r\) |
| *requires* | `file_0_view_0` | `item_ref`, `prerequisite_ref` | \(req_{ij}\) |
| *incompatible* | `file_5_view_0` | `item_a`, `item_b` | \(inc_{ij}\) |
| *bundle* | `file_7_view_0` | `item_a`, `item_b`, `bonus_cents` | \(b_b\), \(a_b\), \(b_b\) |

All resource units are already in the common units (labor = minutes, power = wh, space = ml) due to the unit‑preparation step. No further conversion is performed in the model. The model uses integer quantities for items and binary flags for activation, respecting all constraints described in the problem statement.
**Mathematical Model**

Let  

* \(I\) – set of all item identifiers that appear in the *item* tables.  
* \(C\) – set of all category identifiers that appear in the *category* table.  
* \(R\) – set of all resource identifiers that appear in the *capacity_ledger* and *usage* tables.  
* \(B\) – set of all bundle identifiers \((a,b)\) that appear in the *bundle* table.  
* \(K\) – set of all incompatible pairs \((a,b)\) that appear in the *incompatible* table.  
* \(L\) – set of all prerequisite pairs \((a,b)\) that appear in the *requires* table.  

Parameters:

| Parameter | Definition | Source table & column |
|-----------|------------|-----------------------|
| \(authorized_i\) | 1 if item \(i\) is authorized, 0 otherwise | `item` (file_6_view_0 / file_7_view_0), column `authorized` |
| \(minLot_i\) | Minimum lot size for item \(i\) | `item`, column `minimum_lot` |
| \(maxOrder_i\) | Maximum order size for item \(i\) | `item`, column `maximum_order` |
| \(cat_i\) | Category of item \(i\) | `item`, column `category` |
| \(benefit_i\) | Sum of all `amount_cents` for item \(i\) | `benefit`, column `amount_cents` (aggregate over rows with same `item_ref`) |
| \(fee_i\) | Activation fee for item \(i\) | `item_fee`, column `activation_fee_cents` |
| \(actFee_c\) | Activation fee for category \(c\) | `category`, column `activation_fee_cents` |
| \(minCat_c\) | Minimum quantity for category \(c\) | `category`, column `minimum_quantity` |
| \(maxCat_c\) | Maximum quantity for category \(c\) | `category`, column `maximum_quantity` |
| \(usage_{i,r}\) | Usage of resource \(r\) per unit of item \(i\) | `usage`, columns `amount` (aggregate over rows with same `item_ref` and `resource`) |
| \(cap_r\) | Total available amount of resource \(r\) | `capacity_ledger`, column `amount` (aggregate over rows with same `resource`) |
| \(bonus_{ab}\) | Bonus cents for bundle \((a,b)\) | `bundle`, column `bonus_cents` |
| \(incomp_{ab}\) | Indicator that items \(a,b\) are incompatible | `incompatible`, pair of columns `item_a`, `item_b` |
| \(req_{ab}\) | Indicator that item \(a\) requires item \(b\) | `requires`, pair of columns `item_ref`, `prerequisite_ref` |

Decision variables  

* \(x_i \in \mathbb{Z}_{\ge 0}\) – number of 100 kg units of item \(i\) to produce.  
* \(y_i \in \{0,1\}\) – 1 if item \(i\) is selected (i.e., \(x_i>0\)), 0 otherwise.  
* \(z_c \in \{0,1\}\) – 1 if category \(c\) is used (i.e., any \(x_i\) with \(cat_i=c\) is positive).  
* \(b_{ab} \in \{0,1\}\) – 1 if both items \(a\) and \(b\) of a bundle are selected.  

**Objective**

\[
\max \;\; \sum_{i\in I} \bigl( benefit_i \, x_i - fee_i \, y_i \bigr)
      \;-\; \sum_{c\in C} actFee_c \, z_c
      \;+\; \sum_{(a,b)\in B} bonus_{ab}\, b_{ab}
\]

**Constraints**

1. **Authorization**  
   \[
   x_i \le maxOrder_i \, y_i,\qquad
   x_i \ge minLot_i \, y_i,\qquad
   y_i \le authorized_i,\qquad \forall i\in I
   \]

2. **Category activation**  
   \[
   z_c \ge y_i \quad \forall i\in I:\; cat_i=c
   \]
   \[
   minCat_c \, z_c \;\le\; \sum_{i:cat_i=c} x_i \;\le\; maxCat_c \, z_c,
   \qquad \forall c\in C
   \]

3. **Bundle bonus**  
   \[
   b_{ab} \le y_a,\;\; b_{ab} \le y_b,\;\;
   b_{ab} \ge y_a + y_b - 1,\qquad \forall (a,b)\in B
   \]

4. **Incompatibility**  
   \[
   y_a + y_b \;\le\; 1,\qquad \forall (a,b)\in K
   \]

5. **Prerequisite**  
   \[
   y_a \;\le\; y_b,\qquad \forall (a,b)\in L
   \]

6. **Resource usage**  
   \[
   \sum_{i\in I} usage_{i,r}\, x_i \;\le\; cap_r,\qquad \forall r\in R
   \]

7. **Variable domains**  
   \[
   x_i \in \mathbb{Z}_{\ge 0},\; y_i\in\{0,1\},\;
   z_c\in\{0,1\},\; b_{ab}\in\{0,1\},\qquad \forall i\in I,\; c\in C,\; (a,b)\in B
   \]

---

**Data Mapping Rules**

| Index | Source table(s) | Key column(s) | Notes |
|-------|-----------------|---------------|-------|
| \(I\) | `item` (file_6_view_0, file_7_view_0) | `item_ref` | Union of both tables; each row gives `authorized`, `minimum_lot`, `maximum_order`, `category`. |
| \(C\) | `category` (file_3_view_0) | `category` | Provides `activation_fee_cents`, `minimum_quantity`, `maximum_quantity`. |
| \(R\) | `capacity_ledger` (file_2_view_0) and `usage` (file_11_view_0, file_12_view_0) | `resource` | `capacity_ledger` gives total `amount`; `usage` gives per‑unit `amount`. |
| \(B\) | `bundle` (file_1_view_0) | `item_a`, `item_b` | Provides `bonus_cents`. |
| \(K\) | `incompatible` (file_5_view_0) | `item_a`, `item_b` | Indicates forbidden pairs. |
| \(L\) | `requires` (file_10_view_0) | `item_ref`, `prerequisite_ref` | Indicates prerequisite pairs. |
| \(benefit_i\) | `benefit` (file_0_view_0) | `item_ref`, `amount_cents` | Sum over all rows with same `item_ref`. |
| \(fee_i\) | `item_fee` (file_8_view_0) | `item_ref`, `activation_fee_cents` | One row per item. |
| \(usage_{i,r}\) | `usage` (file_11_view_0, file_12_view_0) | `item_ref`, `resource`, `amount` | Sum over rows with same `item_ref` and `resource`. |
| \(cap_r\) | `capacity_ledger` (file_2_view_0) | `resource`, `amount` | Sum over rows with same `resource`. |

Units match between `usage` and `capacity_ledger`.  The model is an integer linear
program that maximizes net benefit in USD cents subject to the stated
constraints.
**Mathematical Model (symbolic)**  

Let  

* \(I\) – set of item identifiers (from *item* table)  
* \(C\) – set of category identifiers (from *category* table)  
* \(R\) – set of resource identifiers (from *usage* and *capacity_ledger* tables)  
* \(B\subseteq I\times I\) – set of bundle pairs (from *bundle* table)  
* \(K\subseteq I\times I\) – set of incompatible pairs (from *incompatible* table)  
* \(P\subseteq I\times I\) – set of requires pairs (from *requires* table)  

Parameters (all derived from the selected records, i.e. after applying the
“latest‑revision‑≤ 2026‑03‑12, non‑DELETE, deduplication” rule):

| Symbol | Source table & column | Meaning |
|--------|-----------------------|---------|
| \(\text{benefit}_i\) | *benefit* – amount_cents (summed over all components for item \(i\)) | per‑unit signed benefit (cents) |
| \(\text{item\_fee}_i\) | *item_fee* – activation_fee_cents | fee charged once if item \(i\) is ordered |
| \(\text{cat\_fee}_c\) | *category* – activation_fee_cents | fee charged once if any item in category \(c\) is ordered |
| \(\text{cat\_min}_c,\text{cat\_max}_c\) | *category* – minimum_quantity, maximum_quantity | unconditional quantity limits for the category |
| \(\text{auth}_i\) | *item* – authorized | integer flag (0 = unauthorized, >0 = authorized) |
| \(\text{min\_lot}_i,\text{max\_order}_i\) | *item* – minimum_lot, maximum_order | lot size bounds for authorized items |
| \(\text{usage}_{i,r}\) | *usage* – amount (converted to the unit of resource \(r\)) | resource consumption per unit of item \(i\) |
| \(\text{cap}_r\) | *capacity_ledger* – amount (converted to the unit of resource \(r\)) | signed capacity limit for resource \(r\) |
| \(\text{bonus}_{a,b}\) | *bundle* – bonus_cents | bonus (cents) awarded when both items \(a,b\) are ordered |

Decision variables  

* \(x_i \in \mathbb{Z}_{\ge 0}\) – quantity of item \(i\) ordered (integer lot).  
* \(u_i \in \{0,1\}\) – indicator that item \(i\) is ordered (\(u_i=1\) iff \(x_i>0\)).  
* \(y_c \in \{0,1\}\) – indicator that category \(c\) is used.  
* \(w_{a,b} \in \{0,1\}\) – indicator that both items of bundle \((a,b)\) are ordered.  

Auxiliary large constant \(M_i\) (e.g. \(M_i=\text{max\_order}_i\) or a global bound).

**Objective**  
\[
\max \; Z = \sum_{i\in I}\bigl(\text{benefit}_i\,x_i - \text{item\_fee}_i\,u_i\bigr)
          \;-\; \sum_{c\in C}\text{cat\_fee}_c\,y_c
          \;+\; \sum_{(a,b)\in B}\text{bonus}_{a,b}\,w_{a,b}
\]

**Constraints**

1. **Lot bounds (authorized items only)**  
\[
\text{auth}_i\,\text{min\_lot}_i \;\le\; x_i \;\le\; \text{auth}_i\,\text{max\_order}_i
\qquad \forall i\in I
\]

2. **Link \(x_i\) and \(u_i\)**  
\[
x_i \;\le\; M_i\,u_i \qquad
x_i \;\ge\; u_i \quad (\text{if } \text{auth}_i>0) \qquad \forall i\in I
\]

3. **Category activation**  
\[
y_c \;\le\; \sum_{i\in I:\, \text{category}(i)=c} u_i
\qquad \forall c\in C
\]

4. **Category quantity limits (unconditional)**  
\[
\text{cat\_min}_c \;\le\; \sum_{i\in I:\, \text{category}(i)=c} x_i
          \;\le\; \text{cat\_max}_c
\qquad \forall c\in C
\]

5. **Incompatible items**  
\[
u_a + u_b \;\le\; 1 \qquad \forall (a,b)\in K
\]

6. **Requires dependencies**  
\[
u_a \;\le\; u_b \qquad \forall (a,b)\in P
\]

7. **Bundle activation**  
\[
w_{a,b} \;\le\; u_a,\;\; w_{a,b} \;\le\; u_b,\;\;
w_{a,b} \;\ge\; u_a + u_b - 1
\qquad \forall (a,b)\in B
\]

8. **Resource capacity**  
\[
\sum_{i\in I}\text{usage}_{i,r}\,x_i \;\le\; \text{cap}_r
\qquad \forall r\in R
\]

9. **Non‑negativity and integrality**  
\[
x_i \in \mathbb{Z}_{\ge 0},\; u_i,y_c,w_{a,b}\in\{0,1\}
\qquad \forall i\in I,\; c\in C,\; (a,b)\in B
\]

---

### Data‑Mapping Rules (exact source‑to‑parameter linkage)

| Parameter | Source table | Source column | Selection rule applied |
|-----------|--------------|---------------|------------------------|
| \(\text{benefit}_i\) | *benefit* | amount_cents | Sum over all records for the same `item_ref`; only records with `tenant=NORTH`, `effective_date ≤ 2026‑03‑12`, highest `revision`, non‑DELETE, deduped. |
| \(\text{item\_fee}_i\) | *item_fee* | activation_fee_cents | Same selection rule as above. |
| \(\text{cat\_fee}_c\) | *category* | activation_fee_cents | Same selection rule. |
| \(\text{cat\_min}_c,\text{cat\_max}_c\) | *category* | minimum_quantity, maximum_quantity | Same selection rule. |
| \(\text{auth}_i\) | *item* | authorized | Same selection rule. |
| \(\text{min\_lot}_i,\text{max\_order}_i\) | *item* | minimum_lot, maximum_order | Same selection rule. |
| \(\text{usage}_{i,r}\) | *usage* | amount (converted) | Same selection rule; amounts use the unit of resource \(r\). |
| \(\text{cap}_r\) | *capacity_ledger* | amount (converted) | Same selection rule; signed total capacity. |
| \(\text{bonus}_{a,b}\) | *bundle* | bonus_cents | Same selection rule. |
| \(K\) (incompatible pairs) | *incompatible* | item_a, item_b | Same selection rule. |
| \(P\) (requires pairs) | *requires* | item_ref, prerequisite_ref | Same selection rule. |
| \(B\) (bundle pairs) | *bundle* | item_a, item_b | Same selection rule. |

All tables are filtered to `tenant = "NORTH"` and `effective_date ≤ 2026‑03‑12`.  
For each `(tenant, table, record_id)` the record with the highest integer `revision` is kept; if that record’s `action` is `"DELETE"` the record is discarded.  
Identical retransmissions (same `record_id` and `revision`) are counted once.

The model above is a complete, compact symbolic formulation that respects all
requirements of the query and maps every parameter to its exact source data.
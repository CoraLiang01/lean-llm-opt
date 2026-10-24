**Index sets**

- \(I\) – set of item identifiers (item_ref) that survive the selection rule.  
- \(C\) – set of category identifiers that survive the selection rule.  
- \(R\) – set of resource identifiers that appear in the usage or capacity tables.  
- \(B\) – set of bundle identifiers (each bundle is a pair \((a,b)\)).  
- \(P\) – set of incompatible pairs \((a,b)\).  
- \(Q\) – set of requires relations \((i,p)\) meaning item \(i\) requires prerequisite \(p\).

**Parameters (derived from the selected records)**  

| Symbol | Definition (after applying the selection rule) | Source table |
|--------|----------------------------------------------|--------------|
| \(\text{benefit}_i\) | \(\displaystyle\sum_{\substack{r\in\text{benefit}\\ r.\text{item_ref}=i\\ r.\text{effective_date}\le 2026-03-12\\ r.\text{revision}= \max\{rev\}\\ r.\text{action}\neq\text{DELETE}}} r.\text{amount_cents}\) | benefit |
| \(\text{item_fee}_i\) | \(\displaystyle\sum_{\substack{r\in\text{item_fee}\\ r.\text{item_ref}=i\\ r.\text{effective_date}\le 2026-03-12\\ r.\text{revision}= \max\{rev\}\\ r.\text{action}\neq\text{DELETE}}} r.\text{activation_fee_cents}\) | item_fee |
| \(\text{category}_i\) | \(\displaystyle\sum_{\substack{r\in\text{item}\\ r.\text{item_ref}=i\\ r.\text{effective_date}\le 2026-03-12\\ r.\text{revision}= \max\{rev\}\\ r.\text{action}\neq\text{DELETE}}} r.\text{category}\) | item |
| \(\text{authorized}_i\) | \(\displaystyle\sum_{\substack{r\in\text{item}\\ r.\text{item_ref}=i\\ r.\text{effective_date}\le 2026-03-12\\ r.\text{revision}= \max\{rev\}\\ r.\text{action}\neq\text{DELETE}}} r.\text{authorized}\) | item |
| \(\text{min_lot}_i\) | \(\displaystyle\sum_{\substack{r\in\text{item}\\ r.\text{item_ref}=i\\ r.\text{effective_date}\le 2026-03-12\\ r.\text{revision}= \max\{rev\}\\ r.\text{action}\neq\text{DELETE}}} r.\text{minimum_lot}\) | item |
| \(\text{max_order}_i\) | \(\displaystyle\sum_{\substack{r\in\text{item}\\ r.\text{item_ref}=i\\ r.\text{effective_date}\le 2026-03-12\\ r.\text{revision}= \max\{rev\}\\ r.\text{action}\neq\text{DELETE}}} r.\text{maximum_order}\) | item |
| \(\text{usage}_{i,r}\) | \(\displaystyle\sum_{\substack{u\in\text{usage}\\ u.\text{item_ref}=i\\ u.\text{resource}=r\\ u.\text{effective_date}\le 2026-03-12\\ u.\text{revision}= \max\{rev\}\\ u.\text{action}\neq\text{DELETE}}} u.\text{amount}\) | usage (already unit‑converted) |
| \(\text{capacity}_r\) | \(\displaystyle\sum_{\substack{c\in\text{capacity_ledger}\\ c.\text{resource}=r\\ c.\text{effective_date}\le 2026-03-12\\ c.\text{revision}= \max\{rev\}\\ c.\text{action}\neq\text{DELETE}}} c.\text{amount}\) | capacity_ledger |
| \(\text{min}_c\) | \(\displaystyle\sum_{\substack{c\in\text{category}\\ c.\text{category}=c\\ c.\text{effective_date}\le 2026-03-12\\ c.\text{revision}= \max\{rev\}\\ c.\text{action}\neq\text{DELETE}}} c.\text{minimum_quantity}\) | category |
| \(\text{max}_c\) | \(\displaystyle\sum_{\substack{c\in\text{category}\\ c.\text{category}=c\\ c.\text{effective_date}\le 2026-03-12\\ c.\text{revision}= \max\{rev\}\\ c.\text{action}\neq\text{DELETE}}} c.\text{maximum_quantity}\) | category |
| \(\text{act_fee}_c\) | \(\displaystyle\sum_{\substack{c\in\text{category}\\ c.\text{category}=c\\ c.\text{effective_date}\le 2026-03-12\\ c.\text{revision}= \max\{rev\}\\ c.\text{action}\neq\text{DELETE}}} c.\text{activation_fee_cents}\) | category |
| \(\text{bonus}_b\) | \(\displaystyle\sum_{\substack{b\in\text{bundle}\\ b.\text{effective_date}\le 2026-03-12\\ b.\text{revision}= \max\{rev\}\\ b.\text{action}\neq\text{DELETE}}} b.\text{bonus_cents}\) | bundle |
| \(\text{bundle\_items}_b=(a,b)\) | the two item_refs in bundle record \(b\) | bundle |
| \(\text{incomp}_{(a,b)}\) | 1 if pair \((a,b)\) appears in incompatible table after selection | incompatible |
| \(\text{requires}_{(i,p)}\) | 1 if record \((i,p)\) appears in requires table after selection | requires |

All numeric values are in **USD cents**; resource amounts are already converted to the same unit as the capacity ledger (ml, minute, wh).

---

### Decision variables

| Symbol | Type | Meaning |
|--------|------|---------|
| \(q_i\) | integer \(\ge 0\) | quantity of item \(i\) to order (continuous batch equivalents) |
| \(y_i\) | binary | 1 if item \(i\) is ordered (quantity \(>0\)), 0 otherwise |
| \(w_c\) | binary | 1 if any item of category \(c\) is ordered, 0 otherwise |
| \(z_b\) | binary | 1 if both items of bundle \(b\) are ordered, 0 otherwise |

---

### Model

\[
\max \;\; Z = \sum_{i\in I} \text{benefit}_i\,q_i
          \;-\; \sum_{i\in I} \text{item_fee}_i\,y_i
          \;-\; \sum_{c\in C} \text{act_fee}_c\,w_c
          \;+\; \sum_{b\in B} \text{bonus}_b\,z_b
\]

**Item‑lot and authorization**

\[
\begin{aligned}
q_i &\le \text{max_order}_i \; y_i &&\forall i\in I \\
q_i &\ge \text{min_lot}_i \; y_i &&\forall i\in I \\
q_i &\le \text{max_order}_i \; \text{authorized}_i &&\forall i\in I \\
y_i &\le \text{authorized}_i &&\forall i\in I
\end{aligned}
\]

**Incompatibilities**

\[
y_a + y_b \;\le\; 1 &&\forall (a,b)\in P
\]

**Requires**

\[
y_i \;\le\; y_p &&\forall (i,p)\in Q
\]

**Category limits and activation fee**

\[
\begin{aligned}
\sum_{i:\,\text{category}_i=c} q_i &\;\ge\; \text{min}_c &&\forall c\in C \\
\sum_{i:\,\text{category}_i=c} q_i &\;\le\; \text{max}_c &&\forall c\in C \\
w_c &\;\ge\; y_i &&\forall i\in I,\; \text{category}_i=c \\
w_c &\;\le\; \sum_{i:\,\text{category}_i=c} y_i &&\forall c\in C
\end{aligned}
\]

**Bundle bonus**

\[
\begin{aligned}
z_b &\;\le\; y_{a} &&\forall b\in B,\; \text{bundle_items}_b=(a,b) \\
z_b &\;\le\; y_{b} &&\forall b\in B,\; \text{bundle_items}_b=(a,b) \\
z_b &\;\ge\; y_{a}+y_{b}-1 &&\forall b\in B,\; \text{bundle_items}_b=(a,b)
\end{aligned}
\]

**Resource constraints**

\[
\sum_{i\in I} \text{usage}_{i,r}\,q_i \;\le\; \text{capacity}_r
\quad\forall r\in R
\]

**Variable domains**

\[
q_i \in \mathbb{Z}_{\ge 0},\;
y_i \in \{0,1\},\;
w_c \in \{0,1\},\;
z_b \in \{0,1\}
\quad\forall i\in I,\; c\in C,\; b\in B
\]

---

### Data‑mapping rules

1. **Selection rule** – For every source table, keep only records with  
   \(\text{effective_date}\le 2026-03-12\), the highest integer \(\text{revision}\) for each `(tenant, table, record_id)`, and discard those whose `action` is `DELETE`. Identical retransmissions (same hash) are counted once.  
2. **Aggregation** – All parameters above are sums over the selected records.  
3. **Unit conversion** – The `usage` table has already been converted to the same units as the `capacity_ledger` (ml, minute, wh) by the literal‑equivalence rules; no further conversion is applied.  
4. **Index sets** –  
   - \(I\) is the set of `item_ref` values appearing in the selected `item` table.  
   - \(C\) is the set of `category` values appearing in the selected `category` table.  
   - \(R\) is the union of `resource` values appearing in the selected `usage` or `capacity_ledger` tables.  
   - \(B\) is the set of bundle identifiers (each bundle record).  
   - \(P\) is the set of pairs `(item_a,item_b)` from the selected `incompatible` table.  
   - \(Q\) is the set of pairs `(item_ref, prerequisite_ref)` from the selected `requires` table.  

All constraints and the objective use only these mapped parameters and the decision variables defined above. The model is a mixed‑integer linear program that maximizes net benefit (benefit minus fees plus bonuses) while respecting authorization, lot sizing, incompatibilities, requires, category limits, activation fees, bundle bonuses, and resource capacities.
**Index sets**

- \(I\) – set of all item references (from *item* table).  
- \(C\) – set of all categories (from *category* table).  
- \(R\) – set of all resources (from *capacity_ledger* and *usage* tables).  
- \(B\) – set of all bundles (from *bundle* table).  

**Parameters** (all values are taken directly from the corresponding source rows)

| Symbol | Definition | Source |
|--------|------------|--------|
| \(p_i\) | unit benefit in cents for item \(i\) | *item.unit_benefit_cents* |
| \(f_i\) | fixed fee in cents for item \(i\) | *item.item_fee_cents* |
| \(q^{\min}_i\) | minimum lot size for item \(i\) | *item.minimum_lot* |
| \(q^{\max}_i\) | maximum order size for item \(i\) | *item.maximum_order* |
| \(a_i\) | binary flag that item \(i\) is produced (positive quantity) | decision variable |
| \(c_i\) | category of item \(i\) | *item.category* |
| \(A_c\) | activation fee in cents for category \(c\) | *category.activation_fee_cents* |
| \(y_c\) | binary flag that category \(c\) is used | decision variable |
| \(u_{ir}\) | usage of resource \(r\) per unit of item \(i\) | *usage.amount* (unit already matched) |
| \(U_r\) | signed capacity ledger total for resource \(r\) | *capacity_ledger.amount* |
| \(b_{b}\) | bonus in cents for bundle \(b\) | *bundle.bonus_cents* |
| \(i(b,1), i(b,2)\) | the two item references that form bundle \(b\) | *bundle.item_a*, *bundle.item_b* |
| \(r_{ij}\) | prerequisite relation: item \(i\) requires item \(j\) | *requires.item_ref*, *requires.prerequisite_ref* |
| \(I_c\) | \(\{i\in I : c_i=c\}\) | derived from *item.category* |
| \(I_{b,1}\) | item reference \(i(b,1)\) | derived from *bundle* |
| \(I_{b,2}\) | item reference \(i(b,2)\) | derived from *bundle* |

**Decision variables**

- \(x_i \in \mathbb{Z}_{\ge 0}\) – quantity (in 100 kg units) of item \(i\).  
- \(a_i \in \{0,1\}\) – 1 if \(x_i>0\).  
- \(y_c \in \{0,1\}\) – 1 if any item of category \(c\) is produced.  
- \(w_b \in \{0,1\}\) – 1 if both items of bundle \(b\) are produced.  

**Model**

\[
\begin{aligned}
\max\;&
\sum_{i\in I} p_i\,x_i
-\sum_{i\in I} f_i\,a_i
-\sum_{c\in C} A_c\,y_c
+\sum_{b\in B} b_{b}\,w_b
\\[4pt]
\text{s.t.}\;&
\sum_{i\in I} u_{ir}\,x_i \;\le\; U_r
\qquad \forall r\in R
\\[4pt]
&x_i \;\le\; q^{\max}_i\,a_i
\qquad \forall i\in I
\\[4pt]
&x_i \;\ge\; q^{\min}_i\,a_i
\qquad \forall i\in I
\\[4pt]
&a_i \;\le\; \text{authorized}_i
\qquad \forall i\in I
\\[4pt]
&y_c \;\ge\; a_i
\qquad \forall c\in C,\; i\in I_c
\\[4pt]
&y_c \;\le\; \sum_{i\in I_c} a_i
\qquad \forall c\in C
\\[4pt]
&w_b \;\le\; a_{i(b,1)}
\\[4pt]
&w_b \;\le\; a_{i(b,2)}
\\[4pt]
&w_b \;\ge\; a_{i(b,1)} + a_{i(b,2)} - 1
\qquad \forall b\in B
\\[4pt]
&a_i \;\le\; a_j
\qquad \forall (i,j)\in r_{ij}
\\[4pt]
&x_i \in \mathbb{Z}_{\ge 0},\; a_i,w_b,y_c \in \{0,1\}
\qquad \forall i\in I,\; c\in C,\; b\in B
\end{aligned}
\]

**Data‑mapping rules**

1. **Items** – each row of *item* defines a unique \(i\in I\).  
2. **Categories** – each row of *category* defines a unique \(c\in C\).  
3. **Resources** – each distinct *resource* value in *capacity_ledger* or *usage* defines a unique \(r\in R\).  
4. **Bundles** – each row of *bundle* defines a unique \(b\in B\); the two item references are taken from *bundle.item_a* and *bundle.item_b*.  
5. **Requires** – each row of *requires* creates a directed pair \((i,j)\) where \(i\) is *requires.item_ref* and \(j\) is *requires.prerequisite_ref*.  
6. **Usage coefficients** – for each pair \((i,r)\) present in *usage*, set \(u_{ir}\) to the *usage.amount* value; if a pair is missing, \(u_{ir}=0\).  
7. **Capacity limits** – for each resource \(r\), set \(U_r\) to the *capacity_ledger.amount* value (signed).  
8. **Activation fees** – for each category \(c\), set \(A_c\) to *category.activation_fee_cents*.  
9. **Bundle bonuses** – for each bundle \(b\), set \(b_b\) to *bundle.bonus_cents*.  
10. **Authorized flag** – for each item \(i\), set \(\text{authorized}_i=1\) if *item.authorized*="1", else 0.  

All parameters are used exactly as supplied; no additional aggregation or conversion is performed. The model above is a complete, compact symbolic formulation that respects every constraint described in the query.
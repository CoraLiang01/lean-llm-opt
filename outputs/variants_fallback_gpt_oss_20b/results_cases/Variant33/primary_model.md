**Model**

Let  

* \(I\) – set of all item identifiers that appear in the *item* table.  
* \(C\) – set of all category identifiers that appear in the *category* table.  
* \(R\) – set of all resource identifiers that appear in the *capacity_ledger* table.  
* \(B\) – set of all bundle identifiers (pairs \((a,b)\)) that appear in the *bundle* table.  
* \(K\) – set of all incompatible pairs \((a,b)\) that appear in the *incompatible* table.  
* \(P\) – set of all prerequisite pairs \((i,p)\) that appear in the *requires* table.  

**Decision variables**

\[
\begin{aligned}
x_i &\in \mathbb{Z}_{\ge 0} &&\text{(units of item }i\text{ produced)}\\
y_i &\in \{0,1\} &&\text{(1 if }x_i>0\text{)}\\
z_c &\in \{0,1\} &&\text{(1 if any item in category }c\text{ is used)}\\
w_{ab} &\in \{0,1\} &&\text{(1 if both items }a,b\text{ are used)}
\end{aligned}
\qquad i\in I,\;c\in C,\;(a,b)\in B
\]

**Parameters** (all taken directly from the supplied tables)

| Parameter | Definition | Source table & column |
|-----------|------------|-----------------------|
| \(b_i\) | total benefit per unit of item \(i\) (sum of all *amount_cents* rows for that item) | *benefit* – `amount_cents` |
| \(f_i\) | item activation fee for item \(i\) | *item_fee* – `activation_fee_cents` |
| \(a_c\) | category activation fee for category \(c\) | *category* – `activation_fee_cents` |
| \(q^{\min}_i\) | minimum lot for item \(i\) | *item* – `minimum_lot` |
| \(q^{\max}_i\) | maximum order for item \(i\) | *item* – `maximum_order` |
| \(u_i\) | authorization flag for item \(i\) (1 = authorized) | *item* – `authorized` |
| \(c^{\min}_c\) | minimum quantity for category \(c\) | *category* – `minimum_quantity` |
| \(c^{\max}_c\) | maximum quantity for category \(c\) | *category* – `maximum_quantity` |
| \(s_{ir}\) | usage of resource \(r\) per unit of item \(i\) | *usage* – `amount` |
| \(S_r\) | capacity of resource \(r\) | *capacity_ledger* – `amount` |
| \(b_{ab}\) | bonus for bundle \((a,b)\) | *bundle* – `bonus_cents` |
| \(K\) | set of incompatible pairs | *incompatible* – `item_a`,`item_b` |
| \(P\) | set of prerequisite pairs | *requires* – `item_ref`,`prerequisite_ref` |

**Model formulation**

\[
\begin{aligned}
\max\;& \sum_{i\in I} b_i\,x_i
      \;-\;\sum_{i\in I} f_i\,y_i
      \;-\;\sum_{c\in C} a_c\,z_c
      \;+\;\sum_{(a,b)\in B} b_{ab}\,w_{ab} \\[4pt]
\text{s.t. } & 
\begin{cases}
x_i \;\le\; q^{\max}_i\,y_i & \forall i\in I \\[2pt]
x_i \;\ge\; q^{\min}_i\,y_i & \forall i\in I \\[2pt]
y_i \;\le\; u_i & \forall i\in I \\[2pt]
z_c \;\ge\; y_i & \forall c\in C,\; i\in I\text{ with } \text{category}(i)=c \\[2pt]
\sum_{i:\text{cat}(i)=c} x_i \;\ge\; c^{\min}_c & \forall c\in C \\[2pt]
\sum_{i:\text{cat}(i)=c} x_i \;\le\; c^{\max}_c & \forall c\in C \\[2pt]
\sum_{i\in I} s_{ir}\,x_i \;\le\; S_r & \forall r\in R \\[2pt]
y_a + y_b \;\le\; 1 & \forall (a,b)\in K \\[2pt]
y_i \;\le\; y_p & \forall (i,p)\in P \\[2pt]
w_{ab} \;\le\; y_a & \forall (a,b)\in B \\[2pt]
w_{ab} \;\le\; y_b & \forall (a,b)\in B \\[2pt]
w_{ab} \;\ge\; y_a + y_b - 1 & \forall (a,b)\in B \\[2pt]
x_i \in \mathbb{Z}_{\ge 0},\; y_i,z_c,w_{ab}\in\{0,1\} & \forall i\in I,\;c\in C,\;(a,b)\in B
\end{cases}
\end{aligned}
\]

**Data‑mapping rules**

* Every row in the *benefit* table contributes to the parameter \(b_i\) by summing its `amount_cents` values for the same `item_ref`.  
* The *usage* table supplies \(s_{ir}\) for each `(item_ref, resource)` pair; if multiple rows exist for the same pair, their `amount` values are summed.  
* The *capacity_ledger* table supplies \(S_r\) for each `resource`.  
* The *item* table supplies \(q^{\min}_i, q^{\max}_i, u_i\) and the category of each item.  
* The *category* table supplies \(c^{\min}_c, c^{\max}_c, a_c\).  
* The *item_fee* table supplies \(f_i\).  
* The *bundle* table supplies \(b_{ab}\) and defines the set \(B\).  
* The *incompatible* table defines the set \(K\).  
* The *requires* table defines the set \(P\).  

All parameters are taken exactly as they appear in the source tables; no additional aggregation or conversion is performed beyond the unit consistency already ensured by the literal‑query‑unit‑view. The model uses only the data returned in the observation and respects all constraints described in the query.
#### Abstract Mathematical Model

Let:
- $I$ = set of drug types, indexed by $i$, with identifiers ProductName from products.csv.
- $x_i$ = number of units of drug type $i$ to order daily (decision variable), $x_i \in \mathbb{Z}_{\geq 0}$.
- $v_i$ = benefit coefficient of drug type $i$ (Value from products.csv).
- $w_i$ = weight per unit of drug type $i$ (Weight from products.csv).
- $C$ = overall inventory capacity (Capacity from capacity.csv).

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

#### Data Mapping

- $I$ (drug types): file_1_view_0.ProductName
- $v_i$ (benefit): file_1_view_0.Value
- $w_i$ (weight): file_1_view_0.Weight
- $C$ (capacity): file_0_view_0.Capacity
- Decision variables $x_i$ are indexed by file_1_view_0.ProductName

All parameters and index sets are to be taken directly from the specified columns and rows of the returned tables.
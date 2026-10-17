**Mathematical Model**

Let  

* \(I=\{1,\dots,5\}\) be the set of books that satisfy the filter  
  “Product_Name starts with **Books**”.

Parameters (obtained from *file_0_view_0*)  

| Symbol | Meaning | Source column |
|--------|---------|---------------|
| \(R_i\) | Revenue per unit of book \(i\) | `Revenue` |
| \(D_i\) | Demand quantity of book \(i\) | `Demand` |
| \(S_i\) | Initial inventory of book \(i\) | `Initial Inventory` |

Decision variables  

* \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of book \(i\) to be sold.

Objective  

\[
\max \;\sum_{i\in I} R_i\,x_i
\]

Subject to  

\[
\begin{aligned}
x_i &\le D_i &&\forall i\in I &&\text{(cannot sell more than demand)}\\
x_i &\le S_i &&\forall i\in I &&\text{(cannot sell more than available inventory)}\\
x_i &\ge 0 &&\forall i\in I &&\text{(non‑negativity)}\\
x_i &\in \mathbb{Z} &&\forall i\in I &&\text{(integer units)}
\end{aligned}
\]

---

**Data Mapping Rules**

* **Table**: `file_0_view_0` (view of *DifferentStoreSales.csv* after applying the filter).
* **Index set**: `I` is the set of row indices returned by the filter (`Product_Name` prefix `"Books"`).
* **Parameters**:
  * `R_i` ← `Revenue` column of row \(i\).
  * `D_i` ← `Demand` column of row \(i\).
  * `S_i` ← `Initial Inventory` column of row \(i\).
* **Variables**:
  * `x_i` defined for each \(i \in I\) as an integer, non‑negative decision variable.

No other data or constraints are introduced beyond those explicitly stated in the query.
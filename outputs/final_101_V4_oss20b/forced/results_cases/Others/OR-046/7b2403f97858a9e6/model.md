**Mathematical Model**

Let  

* \(I=\{1,\dots ,10\}\) be the set of products.  
* \(x_i\) – number of units of product \(i\) ordered each day.  

Parameters (obtained from the supplied tables)

| Parameter | Source | Description |
|-----------|--------|-------------|
| \(w_i\) | `file_1_view_0` – column **Weight** | Weight (kg) of one unit of product \(i\). |
| \(v_i\) | `file_1_view_0` – column **Value** | Profit (yuan) per unit of product \(i\). |
| \(C\) | `file_0_view_0` – column **Capacity** | Total weight capacity (kg) available each day. |

**Objective**

\[
\max \;\; \sum_{i\in I} v_i\,x_i
\]

**Constraints**

\[
\sum_{i\in I} w_i\,x_i \;\le\; C
\]

\[
x_i \;\ge\; 0,\qquad x_i \in \mathbb{Z}\;\;\forall i\in I
\]

---

**Data Mapping**

* `file_0_view_0`  
  * Column **Capacity** → parameter \(C\).

* `file_1_view_0`  
  * Column **Weight** → parameter \(w_i\) for each row \(i\).  
  * Column **Value** → parameter \(v_i\) for each row \(i\).  
  * Column **ProductName** is used only for identification; it does not appear in the model.

No other tables or constraints are referenced. The model uses all rows returned from the two tables, as required by the query.
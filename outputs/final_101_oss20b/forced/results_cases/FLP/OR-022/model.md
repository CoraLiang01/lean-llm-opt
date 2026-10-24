**Mathematical Model**

Let  

* \(I\) be the set of products whose *Product Name* contains the string **“27in”** (obtained from the filtered view `file_0_view_0`).  
* For each \(i\in I\) define the parameters  

\[
\begin{aligned}
\text{rev}_i &\;=\;\text{Revenue of product }i,\\
\text{dem}_i &\;=\;\text{Demand of product }i,\\
\text{inv}_i &\;=\;\text{Initial Inventory of product }i .
\end{aligned}
\]

* Decision variable  

\[
x_i \;\ge\; 0 \qquad \forall i\in I
\]

(number of units of product \(i\) to be sold).

---

**Objective**

\[
\max \;\sum_{i\in I} \text{rev}_i \, x_i
\]

---

**Constraints**

\[
\begin{aligned}
x_i &\;\le\; \text{dem}_i &&\forall i\in I \quad (\text{cannot sell more than demand})\\[4pt]
x_i &\;\le\; \text{inv}_i &&\forall i\in I \quad (\text{cannot sell more than inventory})\\[4pt]
x_i &\;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(I\) (index set) | `file_0_view_0` | `Product Name` (filtered where contains “27in”) |
| \(\text{rev}_i\) | `file_0_view_0` | `Revenue` |
| \(\text{dem}_i\) | `file_0_view_0` | `Demand` |
| \(\text{inv}_i\) | `file_0_view_0` | `Initial Inventory` |

The filter condition applied to the source table is:

```
WHERE "Product Name" CONTAINS '27in'
```

No other data or constraints are introduced beyond those explicitly required by the query.
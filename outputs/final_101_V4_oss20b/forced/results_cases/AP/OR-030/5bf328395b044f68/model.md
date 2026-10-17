**Mathematical Model**

Let  

* \(I=\{1163,1501,1576,1793,2438,4297,4806,4942\}\) – the set of record identifiers for the
  *FDK57* car models that satisfy the filter in the query.  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**)  
  * \(d_i\) – demand quantity (from column **Demand**)  
  * \(v_i\) – initial inventory (from column **Initial Inventory**)  
  * \(x_i\) – quantity of model \(i\) to be sold (decision variable)

---

### Objective
\[
\max \;\; \sum_{i\in I} r_i\,x_i
\]

### Constraints
\[
\begin{aligned}
&0 \;\le\; x_i \;\le\; d_i &&\forall i\in I \quad &&\text{(cannot sell more than demand)}\\[4pt]
&0 \;\le\; x_i \;\le\; v_i &&\forall i\in I \quad &&\text{(cannot sell more than inventory)}\\[4pt]
&x_i \in \mathbb{Z}_+ &&\forall i\in I \quad &&\text{(integer units)}
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` |
| \(v_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |
| \(I\) | Record identifiers | `file_0_view_0` | `source_row` |

*Each record in the returned rows of `file_0_view_0` supplies the corresponding values for \(r_i\), \(d_i\), and \(v_i\). No additional numerical values are introduced beyond those present in the source data.*
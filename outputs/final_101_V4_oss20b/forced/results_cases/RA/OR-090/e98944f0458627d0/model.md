**Symbolic Model**

Let  

* \(I=\{\,\text{product}\,\}\) index set of the 100 products (from *factory_products_100.csv*).  
* \(R=\{\,\text{resource}\,\}\) index set of the 3 resources (from *resources_capacities.csv*).  

Parameters  

\[
\begin{aligned}
p_i      &\;=\;\text{profit per unit of product }i,\\
r_{1,i}  &\;=\;\text{R1 consumption per unit of product }i,\\
r_{2,i}  &\;=\;\text{R2 consumption per unit of product }i,\\
r_{3,i}  &\;=\;\text{R3 consumption per unit of product }i,\\
d_i      &\;=\;\text{upper demand (units) for product }i,\\
b_i      &\;=\;\text{batch size (units) for product }i\;(=10),\\
c_k      &\;=\;\text{total capacity of resource }k.
\end{aligned}
\]

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad\forall i\in I
\]

( number of batches of product \(i\) to produce).

---

**Objective**

\[
\max \;\sum_{i\in I} 10\,p_i\,x_i
\]

---

**Constraints**

1. **Resource limits**  
   \[
   \sum_{i\in I} 10\,r_{k,i}\,x_i \;\le\; c_k
   \qquad\forall k\in R
   \]

2. **Demand limits**  
   \[
   10\,x_i \;\le\; d_i
   \qquad\forall i\in I
   \]

3. **Non‑negativity & integrality**  
   \[
   x_i \in \mathbb{Z}_{\ge 0}\qquad\forall i\in I
   \]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | *factory_products_100.csv* | `product` |
| \(R\) | *resources_capacities.csv* | `resource` |
| \(p_i\) | `file_0_view_0` | `profit_per_unit` |
| \(r_{1,i}\) | `file_0_view_0` | `r1_per_unit` |
| \(r_{2,i}\) | `file_0_view_0` | `r2_per_unit` |
| \(r_{3,i}\) | `file_0_view_0` | `r3_per_unit` |
| \(d_i\) | `file_0_view_0` | `upper_demand_units` |
| \(b_i\) | `file_0_view_0` | `batch_size_units` |
| \(c_k\) | `file_1_view_0` | `capacity` |

All parameters are taken directly from the corresponding CSV records; no values are hard‑coded in the model.
**Mathematical Model**

Let  

- \(I\) be the set of all products (indexed by the value in the *Product Name* column).  

Parameters (all read from the table `file_0_view_0`):  

\[
\begin{aligned}
r_i &\;=\; \text{Revenue of product } i, \\
d_i &\;=\; \text{Demand of product } i, \\
inv_i &\;=\; \text{Initial Inventory of product } i .
\end{aligned}
\]

Decision variable  

\[
x_i \;\ge\; 0 \quad\text{(units of product } i \text{ to sell)} .
\]

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
&0 \;\le\; x_i \;\le\; \min\{d_i,\;inv_i\} &&\forall i\in I .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column Name |
|-----------|--------------|-------------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(inv_i\) | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) is obtained from the distinct values in the `Product Name` column of `file_0_view_0`. No other data are required.
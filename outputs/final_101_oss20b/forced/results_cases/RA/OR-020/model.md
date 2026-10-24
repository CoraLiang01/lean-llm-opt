**Mathematical Model**

Let  

\[
I=\{\,i\mid i \text{ is a product name in the data}\,\}
\]

\[
\begin{aligned}
\text{Parameters:}\quad
&\text{Revenue}_i &&\text{(revenue per unit of product }i)\\
&\text{Demand}_i   &&\text{(maximum units demanded for product }i)\\
&\text{Inventory}_i&&\text{(initial inventory of product }i)
\end{aligned}
\]

\[
\text{Decision variables:}\quad
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

---

**Objective**

\[
\max \;\sum_{i\in I}\text{Revenue}_i \; x_i
\]

---

**Constraints**

\[
\begin{aligned}
&x_i \;\le\; \text{Demand}_i &&\forall i\in I \quad\text{(cannot sell more than demand)}\\[4pt]
&x_i \;\le\; \text{Inventory}_i &&\forall i\in I \quad\text{(cannot sell more than available stock)}\\[4pt]
&x_i \;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column Name |
|--------|---------------|-------------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(\text{Revenue}_i\) | `file_0_view_0` | `Revenue` |
| \(\text{Demand}_i\) | `file_0_view_0` | `Demand` |
| \(\text{Inventory}_i\) | `file_0_view_0` | `Initial Inventory` |
| \(x_i\) | – | – |

All parameters are real numbers; decision variables \(x_i\) are non‑negative integers.
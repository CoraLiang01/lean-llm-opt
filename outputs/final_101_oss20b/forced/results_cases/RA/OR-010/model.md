**Mathematical Model**

Let  

\[
I=\{i\mid \text{product }i\text{ appears in the dataset}\}
\]

\[
\begin{aligned}
\text{Parameters:}\quad
& R_i &&\text{revenue per unit of product }i\\
& D_i &&\text{demand for product }i\\
& I_i &&\text{initial inventory of product }i
\end{aligned}
\]

\[
\begin{aligned}
\text{Decision variables:}\quad
& x_i &&\in \mathbb{Z}_{\ge 0}\quad\text{(units of product }i\text{ to fulfill)}
\end{aligned}
\]

\[
\begin{aligned}
\max\;& \sum_{i\in I} R_i\,x_i \\
\text{s.t.}\;& 0 \le x_i \le D_i &&\forall i\in I \\
& 0 \le x_i \le I_i &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(R_i\) | `file_0_view_0` | `Revenue` |
| \(D_i\) | `file_0_view_0` | `Demand` |
| \(I_i\) | `file_0_view_0` | `Initial Inventory` |
| \(x_i\) | – | – |

All parameters are taken directly from the specified columns of the table `file_0_view_0`. The decision variables \(x_i\) are integer and non‑negative as required.
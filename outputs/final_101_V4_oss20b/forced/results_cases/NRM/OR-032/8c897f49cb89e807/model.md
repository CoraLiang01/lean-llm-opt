**Symbolic Model**

Let  

* \(I\) – index set of all books products extracted from the view  
* \(r_i\) – revenue per unit of product \(i\)  
* \(d_i\) – deterministic demand for product \(i\)  
* \(I_i\) – initial inventory of product \(i\)  
* \(x_i\) – number of units of product \(i\) to be fulfilled  

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le I_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_{+} && \forall i\in I
\end{aligned}
\]

(If the problem statement requires strictly positive units, replace \(0\le x_i\) by \(1\le x_i\).)

---

**Data Mapping**

| Parameter | Source | Table ID | Column Name |
|-----------|--------|----------|-------------|
| \(r_i\) | revenue | `file_0_view_0` | `Revenue` |
| \(d_i\) | demand | `file_0_view_0` | `Demand` |
| \(I_i\) | initial inventory | `file_0_view_0` | `Initial Inventory` |
| \(I\) | index set | `file_0_view_0` | `Product_Name` |

All parameter values are taken directly from the corresponding columns of the view `file_0_view_0`; no additional data or transformations are introduced.
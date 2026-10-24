**Mathematical model**

Let  

* \(I=\{i\mid i=\text{ProductName in }file\_1\_view\_0\}\)  
* \(x_i\ge 0\) quantity of product \(i\) ordered per day  

Parameters (all taken directly from the source tables)

| Parameter | Source table | Source column | Meaning |
|-----------|--------------|---------------|---------|
| \(v_i\)   | file\_1\_view\_0 | Value | profit per unit of product \(i\) |
| \(w_i\)   | file\_1\_view\_0 | Weight | inventory weight per unit of product \(i\) |
| \(C\)     | file\_0\_view\_0 | Capacity | total available inventory capacity |

The optimization problem is

\[
\begin{aligned}
\max_{x}\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t. }& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0 \qquad \forall i\in I
\end{aligned}
\]

---

**Data‑mapping rules**

* **Index set \(I\)** – each distinct `ProductName` row in `file_1_view_0` becomes an index element \(i\).  
* **Parameter \(v_i\)** – map the `Value` column of `file_1_view_0` to \(v_i\) for the corresponding \(i\).  
* **Parameter \(w_i\)** – map the `Weight` column of `file_1_view_0` to \(w_i\) for the corresponding \(i\).  
* **Parameter \(C\)** – map the single `Capacity` value in `file_0_view_0` to \(C\).  

No additional constraints, variables, or objective terms are introduced beyond those specified in the query.
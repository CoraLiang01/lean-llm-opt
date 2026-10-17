**Mathematical Model**

Let  

- \(I\) be the set of products whose *Product Name* contains the string “Baby”.  
- For each \(i\in I\)  
  - \(r_i\) : revenue per unit (from column **Revenue**)  
  - \(d_i\) : demand quantity (from column **Demand**)  
  - \(v_i\) : initial inventory (from column **Initial Inventory**)  
  - \(x_i\) : number of units of product \(i\) to be sold (decision variable)

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le v_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_+ && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Variable | Source Table | Column(s) | Mapping Rule |
|----------------------|--------------|-----------|--------------|
| \(I\) | `file_0_view_0` | *Product Name* (filtered by contains “Baby”) | Each row satisfying the filter defines one index \(i\). |
| \(r_i\) | `file_0_view_0` | `Revenue` | \(r_i\) is the value in the **Revenue** column for row \(i\). |
| \(d_i\) | `file_0_view_0` | `Demand` | \(d_i\) is the value in the **Demand** column for row \(i\). |
| \(v_i\) | `file_0_view_0` | `Initial Inventory` | \(v_i\) is the value in the **Initial Inventory** column for row \(i\). |
| \(x_i\) | Decision variable | – | Decision variable defined for each \(i\in I\). |

*All numeric values are taken directly from the corresponding columns of the filtered view `file_0_view_0`.*
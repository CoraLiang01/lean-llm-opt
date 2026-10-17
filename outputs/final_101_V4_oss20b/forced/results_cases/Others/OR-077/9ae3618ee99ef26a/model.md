**Mathematical Model**

Let  

\[
I=\{1,2,\dots ,140\}
\]

\[
\begin{aligned}
\text{maximize}\quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{subject to}\quad & \sum_{i\in I} w_i\,x_i \;\le\; 15 \\[4pt]
& x_i \in \{0,1\}\qquad \forall i\in I
\end{aligned}
\]

where  

* \(v_i\) – value of item \(i\) (RMB)  
* \(w_i\) – weight of item \(i\) (kg)  
* \(x_i\) – binary decision variable (1 if item \(i\) is selected, 0 otherwise).

---

### Data Mapping

| Index set | Source | Column | Mapping |
|-----------|--------|--------|---------|
| \(I\) | `file_0_view_0` | `item` | All rows (1–140) |
| \(v_i\) | `file_0_view_0` | `value` | Parameter value for each item |
| \(w_i\) | `file_0_view_0` | `weight` | Parameter weight for each item |

No filtering is applied; all 140 items are considered. The capacity constraint uses the constant 15 kg.
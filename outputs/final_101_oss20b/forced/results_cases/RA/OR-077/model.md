**Mathematical Model**

Let  

- \(I=\{i\mid i\text{ is an item in table }file\_0\_view\_0\}\)  
- \(x_i\in\{0,1\}\) (1 if item \(i\) is displayed, 0 otherwise)  

Parameters  

- \(v_i\) – value of item \(i\)  
- \(w_i\) – weight of item \(i\)  
- \(C=15\) – total weight capacity of the merchandise counter  

\[
\begin{aligned}
\max\;&\sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;&\sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
&x_i\in\{0,1\}\qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(v_i\)   | `file_0_view_0` | `value` |
| \(w_i\)   | `file_0_view_0` | `weight` |
| \(C\)     | — (constant) | 15 |

The index set \(I\) is obtained from the `item` column of `file_0_view_0`.
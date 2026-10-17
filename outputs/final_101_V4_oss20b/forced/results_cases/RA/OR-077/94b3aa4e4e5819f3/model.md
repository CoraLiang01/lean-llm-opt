**Mathematical Model**

Let  

* \(I\) be the set of items indexed by the value of the column **item** in  
  `file_0_view_0`.  
* For each \(i\in I\) let  

  \[
  v_i \;=\; \text{value of item }i \quad(\text{column } \texttt{value})
  \]
  \[
  w_i \;=\; \text{weight of item }i \quad(\text{column } \texttt{weight})
  \]

* Decision variable  

  \[
  x_i \in \{0,1\}\qquad \forall i\in I
  \]

* Capacity constant  

  \[
  W = 15
  \]

The optimization problem is

\[
\begin{aligned}
\max_{x}\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; W \\[4pt]
& x_i \in \{0,1\}\qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Index set | `file_0_view_0` | `item` |
| \(v_i\) | Parameter | `file_0_view_0` | `value` |
| \(w_i\) | Parameter | `file_0_view_0` | `weight` |
| \(x_i\) | Decision variable | – | – |
| \(W\) | Constant | – | – |

All parameters are taken directly from the rows of `file_0_view_0`; no values are enumerated in the model. The capacity constant \(W=15\) is supplied by the problem statement.
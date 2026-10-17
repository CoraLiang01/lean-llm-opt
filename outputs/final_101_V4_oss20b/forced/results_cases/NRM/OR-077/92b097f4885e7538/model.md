**Mathematical Model**

Let  

- \(I\) be the set of items indexed by the values in column **item** of table *file_0_view_0*.  
- For each \(i\in I\)  
  - \(v_i\) = value of item \(i\) (column **value**)  
  - \(w_i\) = weight of item \(i\) (column **weight**)  
- \(W = 15\) (total weight capacity).  

Decision variable  

- \(x_i \in \{0,1\}\) – 1 if item \(i\) is selected, 0 otherwise.  

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} w_i\,x_i \;\le\; W \\[4pt]
& x_i \in \{0,1\}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | *file_0_view_0* | **item** |
| \(v_i\) | *file_0_view_0* | **value** |
| \(w_i\) | *file_0_view_0* | **weight** |

The capacity \(W\) is a constant defined in the problem statement (15).
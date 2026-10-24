**Mathematical Model**

Let  

- \(I\) be the set of course identifiers that belong to the *Operations Research* discipline (obtained from the filtered view `file_0_view_0`).  

Parameters  

- \(\text{credits}_i\) – number of credits of course \(i\).  
- \(\text{interest}_i\) – interest‑points score of course \(i\).  

Decision variables  

- \(x_i \in \{0,1\}\) – 1 if course \(i\) is selected, 0 otherwise.  

\[
\begin{aligned}
\max\;& \sum_{i\in I} \text{interest}_i \, x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} \text{credits}_i \, x_i \;\le\; 20 \\[4pt]
& x_i \in \{0,1\}, \qquad \forall i\in I .
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Variable | Source Table | Column |
|----------------------|--------------|--------|
| \(\text{credits}_i\) | `file_0_view_0` | `credits` |
| \(\text{interest}_i\) | `file_0_view_0` | `interest_points` |
| \(x_i\) | – | – (binary decision variable) |

The index set \(I\) is the collection of `course_id` values present in `file_0_view_0`. No other data are required.
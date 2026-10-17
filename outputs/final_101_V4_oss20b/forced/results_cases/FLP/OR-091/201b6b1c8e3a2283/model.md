**Mathematical Model**

Let  

* \(I\) – set of Operations Research courses (identified by `course_id`).  
* \(x_i \in \{0,1\}\) – binary variable, 1 if course \(i\) is selected, 0 otherwise.  

Parameters (indexed by \(i\in I\))  

* \(\text{credits}_i\) – credit value of course \(i\).  
* \(\text{interest}_i\) – interest‑points score of course \(i\).  

---

\[
\begin{aligned}
\max\;& \sum_{i\in I} \text{interest}_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} \text{credits}_i\,x_i \;\le\; 20 \\[4pt]
& x_i \in \{0,1\}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(\text{credits}_i\) | Course credits | `file_0_view_0` | `credits` |
| \(\text{interest}_i\) | Interest points | `file_0_view_0` | `interest_points` |
| \(I\) | Course identifiers | `file_0_view_0` | `course_id` |

The table `file_0_view_0` contains only rows where `discipline = "Operations Research"`. Each row supplies the values for the corresponding course in the index set \(I\). No other data or constraints are required.
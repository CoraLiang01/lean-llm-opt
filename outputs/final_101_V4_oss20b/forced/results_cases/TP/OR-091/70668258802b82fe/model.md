**Mathematical Model**

Let  

* \(I\) – set of Operations Research courses (indices are the `course_id` values).  
* For each \(i\in I\)  
  * \(c_i\) – credits of course \(i\) (parameter).  
  * \(p_i\) – interest‑points of course \(i\) (parameter).  
* Decision variable  
  * \(x_i \in \{0,1\}\) – 1 if course \(i\) is selected, 0 otherwise.

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} c_i\,x_i \;\le\; 20 \\[4pt]
& x_i \in \{0,1\}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter / Variable | Source | Table ID | Column(s) | Mapping |
|----------------------|--------|----------|-----------|---------|
| \(I\) | Course identifiers | `file_0_view_0` | `course_id` | Each distinct `course_id` in the filtered view forms an index in \(I\). |
| \(c_i\) | Credits | `file_0_view_0` | `credits` | For each row, assign the numeric value of `credits` to \(c_i\). |
| \(p_i\) | Interest points | `file_0_view_0` | `interest_points` | For each row, assign the numeric value of `interest_points` to \(p_i\). |
| \(x_i\) | Decision variable | – | – | Binary variable indexed by \(i\in I\). |

*The filtered view `file_0_view_0` contains only rows where `discipline = "Operations Research"`. No other data is required for this model.*
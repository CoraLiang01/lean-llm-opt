**Mathematical Model**

Let  

- \(I=\{1,2,\dots ,140\}\) be the set of items.  
- \(v_i\) = value of item \(i\) (RMB).  
- \(w_i\) = weight of item \(i\) (kg).  
- \(x_i\in\{0,1\}\) = 1 if item \(i\) is selected, 0 otherwise.  

\[
\begin{aligned}
\max\;& Z = \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; 15 \\[4pt]
& x_i \in \{0,1\}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source file | Column in file |
|-----------|-------------|----------------|
| \(v_i\)   | `Test_Dataset/Large-scale-or/Others_example/Others3/value.csv` | `value` |
| \(w_i\)   | `Test_Dataset/Large-scale-or/Others_example/Others3/value.csv` | `weight` |
| Capacity | – | 15 (fixed in the model) |

The index set \(I\) is derived from all rows of the CSV file; each row provides a unique item identifier (the row number). No additional filtering or aggregation is applied.
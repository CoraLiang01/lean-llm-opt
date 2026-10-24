**Mathematical Model**

Let  

* \(F\) – set of all foods (identifiers from the column **Food**).  
* Decision variable  
  \[
  x_f \ge 0 \qquad \forall f\in F
  \]
  (number of servings of food \(f\)).  

Parameters (obtained from the CSV file *cost.csv*)  

\[
\begin{aligned}
c_f      &\;=\;\text{Cost per serving of food }f,\\
\text{Cal}_f &\;=\;\text{Calories per serving of food }f,\\
\text{Prot}_f &\;=\;\text{Protein (g) per serving of food }f,\\
\text{Fat}_f  &\;=\;\text{Fat (g) per serving of food }f,\\
\text{VitC}_f &\;=\;\text{Vitamin C (mg) per serving of food }f.
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{f\in F} c_f\,x_f
\]

**Constraints**

\[
\begin{aligned}
\sum_{f\in F} \text{Cal}_f\,x_f &\;\ge\; 2000 &&\text{(Calories)}\\[4pt]
\sum_{f\in F} \text{Prot}_f\,x_f &\;\ge\; 50   &&\text{(Protein)}\\[4pt]
\sum_{f\in F} \text{VitC}_f\,x_f &\;\ge\; 60   &&\text{(Vitamin C)}\\[4pt]
\sum_{f\in F} \text{Fat}_f\,x_f  &\;\le\; 70   &&\text{(Fat)}
\end{aligned}
\]

All variables \(x_f\) are continuous and non‑negative.

---

**Data Mapping**

| Parameter | Source Table | Column in Table |
|-----------|--------------|-----------------|
| \(c_f\) | `file_0_view_0` | `Cost` |
| \(\text{Cal}_f\) | `file_0_view_0` | `Calories` |
| \(\text{Prot}_f\) | `file_0_view_0` | `Protein(g)` |
| \(\text{Fat}_f\) | `file_0_view_0` | `Fat(g)` |
| \(\text{VitC}_f\) | `file_0_view_0` | `VitaminC(mg)` |

The index set \(F\) is the set of all distinct values appearing in the **Food** column of the same table. No other data or constraints are introduced.
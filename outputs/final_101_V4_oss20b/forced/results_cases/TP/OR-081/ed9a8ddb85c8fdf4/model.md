**Mathematical Model**

Let  

* \(F\) – set of all foods (identifiers from the *Food* column).  
* For each \(f\in F\)  
  * \(c_f\) – calories per serving  
  * \(p_f\) – protein (g) per serving  
  * \(f_f\) – fat (g) per serving  
  * \(v_f\) – vitamin C (mg) per serving  
  * \(k_f\) – cost (USD) per serving  

Decision variable  

\[
x_f \;\ge 0 \qquad\text{(number of servings of food }f\text{)}
\]

Objective  

\[
\min \;\sum_{f\in F} k_f\,x_f
\]

Constraints  

\[
\begin{aligned}
\sum_{f\in F} c_f\,x_f &\;\ge\; 2000 &&\text{(calories)}\\[4pt]
\sum_{f\in F} p_f\,x_f &\;\ge\; 50   &&\text{(protein)}\\[4pt]
\sum_{f\in F} v_f\,x_f &\;\ge\; 60   &&\text{(vitamin C)}\\[4pt]
\sum_{f\in F} f_f\,x_f &\;\le\; 70   &&\text{(fat)}
\end{aligned}
\]

All variables \(x_f\) are continuous and non‑negative.

---

**Data Mapping Rules**

| Parameter | Source Table | Column Name | Mapping |
|-----------|--------------|-------------|---------|
| \(c_f\) | `file_0_view_0` | `Calories` | `c_f = Calories` |
| \(p_f\) | `file_0_view_0` | `Protein(g)` | `p_f = Protein(g)` |
| \(f_f\) | `file_0_view_0` | `Fat(g)` | `f_f = Fat(g)` |
| \(v_f\) | `file_0_view_0` | `VitaminC(mg)` | `v_f = VitaminC(mg)` |
| \(k_f\) | `file_0_view_0` | `Cost` | `k_f = Cost` |
| Index set \(F\) | `file_0_view_0` | `Food` | `F = {Food}` |

All numeric values are taken directly from the corresponding columns; the identifiers in the *Food* column define the index set. No additional constraints or bounds are imposed beyond those listed in the model.
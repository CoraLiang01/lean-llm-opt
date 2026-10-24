**Mathematical model**

Let  

* \(F\) – set of all foods (identifiers from *cost.csv*).  
* For each \(f\in F\)  
  * \(c_f\) – calories per serving  
  * \(p_f\) – protein (g) per serving  
  * \(t_f\) – fat (g) per serving  
  * \(v_f\) – vitamin C (mg) per serving  
  * \(k_f\) – cost (USD) per serving  

Decision variable  

\[
x_f \;\ge\; 0 \qquad\text{(number of servings of food }f\text{)}
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
\sum_{f\in F} t_f\,x_f &\;\le\; 70   &&\text{(fat)}
\end{aligned}
\]

All variables \(x_f\) are continuous and non‑negative.

---

**Data‑mapping rules**

| Parameter | Source column | Table ID | Mapping expression |
|-----------|---------------|----------|---------------------|
| \(F\) | `Food` | `file_0_view_0` | set of all distinct `Food` values |
| \(c_f\) | `Calories` | `file_0_view_0` | `c_f = Calories` |
| \(p_f\) | `Protein(g)` | `file_0_view_0` | `p_f = Protein(g)` |
| \(t_f\) | `Fat(g)` | `file_0_view_0` | `t_f = Fat(g)` |
| \(v_f\) | `VitaminC(mg)` | `file_0_view_0` | `v_f = VitaminC(mg)` |
| \(k_f\) | `Cost` | `file_0_view_0` | `k_f = Cost` |

No additional constraints or bounds are required beyond those listed in the model.
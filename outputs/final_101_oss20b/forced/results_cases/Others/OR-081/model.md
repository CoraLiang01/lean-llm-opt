**Mathematical Model**

Let  

* \(F\) – set of all foods listed in *cost.csv* (120 items).  
* For each \(f\in F\)  
  * \(c_f\) – calories per serving (kcal)  
  * \(p_f\) – protein per serving (g)  
  * \(f_f\) – fat per serving (g)  
  * \(v_f\) – vitamin C per serving (mg)  
  * \(w_f\) – cost per serving (USD)  

Decision variable  

\[
x_f \;\ge\; 0 \qquad\text{(servings of food }f\text{)}
\]

Objective  

\[
\min \;\sum_{f\in F} w_f\,x_f
\]

Nutrition constraints  

\[
\begin{aligned}
\sum_{f\in F} c_f\,x_f &\;\ge\; 2000 &&\text{(Calories)}\\[4pt]
\sum_{f\in F} p_f\,x_f &\;\ge\; 50   &&\text{(Protein)}\\[4pt]
\sum_{f\in F} v_f\,x_f &\;\ge\; 60   &&\text{(Vitamin C)}\\[4pt]
\sum_{f\in F} f_f\,x_f &\;\le\; 70   &&\text{(Fat)}
\end{aligned}
\]

All variables \(x_f\) are continuous (fractional servings allowed).

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(c_f\) | Calories | file_0_view_0 | `Calories` |
| \(p_f\) | Protein | file_0_view_0 | `Protein(g)` |
| \(f_f\) | Fat | file_0_view_0 | `Fat(g)` |
| \(v_f\) | Vitamin C | file_0_view_0 | `VitaminC(mg)` |
| \(w_f\) | Cost | file_0_view_0 | `Cost` |

*Index set*  
\(F = \{\,\text{Food}\,\}\) taken from all rows returned by the query (120 foods).

*Variable domain*  
\(x_f \in \mathbb{R}_{\ge 0}\) for all \(f\in F\).

*Constraints*  
Each nutrition constraint is a linear inequality over the variables \(x_f\) with coefficients taken directly from the corresponding columns in the source table.

No additional bounds or integrality requirements are imposed.
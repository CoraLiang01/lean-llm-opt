**Mathematical Model**

Let  

- \(F\) be the set of all foods listed in *cost.csv* (120 items).  
- For each \(f\in F\) let  

\[
\begin{aligned}
c_f   &\;=\;\text{Calories per serving of }f,\\
p_f   &\;=\;\text{Protein (g) per serving of }f,\\
t_f   &\;=\;\text{Fat (g) per serving of }f,\\
v_f   &\;=\;\text{Vitamin C (mg) per serving of }f,\\
\text{cost}_f &\;=\;\text{Cost (USD) per serving of }f .
\end{aligned}
\]

Decision variables  

\[
x_f \;\ge\;0 \qquad (f\in F)
\]

(number of servings of food \(f\); fractional servings allowed).

**Objective**

\[
\min \;\sum_{f\in F}\text{cost}_f\,x_f
\]

**Constraints**

\[
\begin{aligned}
\sum_{f\in F} c_f\,x_f &\;\ge\; 2000 &&\text{(Calories)}\\[4pt]
\sum_{f\in F} p_f\,x_f &\;\ge\; 50   &&\text{(Protein)}\\[4pt]
\sum_{f\in F} v_f\,x_f &\;\ge\; 60   &&\text{(Vitamin C)}\\[4pt]
\sum_{f\in F} t_f\,x_f &\;\le\; 70   &&\text{(Fat)}
\end{aligned}
\]

All variables \(x_f\) are continuous and non‑negative.

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(c_f\)   | Calories | file_0_view_0 | `Calories` |
| \(p_f\)   | Protein(g) | file_0_view_0 | `Protein(g)` |
| \(t_f\)   | Fat(g) | file_0_view_0 | `Fat(g)` |
| \(v_f\)   | VitaminC(mg) | file_0_view_0 | `VitaminC(mg)` |
| \(\text{cost}_f\) | Cost | file_0_view_0 | `Cost` |

The index set \(F\) is the set of all distinct values in the `Food` column of *cost.csv* (120 foods). No additional filtering or aggregation is applied. All numeric columns are interpreted as real numbers; the `Food` column provides the identifiers for the index set. The model uses the full dataset as supplied.
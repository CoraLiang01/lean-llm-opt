**Mathematical Model**

Let  

- \(F\) be the set of all foods (identifier given by the column **Food**).  
- For each \(f\in F\) define the parameters  

\[
\begin{aligned}
c_f   &:= \text{Calories of food }f \quad (\text{column } \texttt{Calories})\\
p_f   &:= \text{Protein (g) of food }f \quad (\text{column } \texttt{Protein(g)})\\
f_f   &:= \text{Fat (g) of food }f \quad (\text{column } \texttt{Fat(g)})\\
v_f   &:= \text{Vitamin C (mg) of food }f \quad (\text{column } \texttt{VitaminC(mg)})\\
\text{cost}_f &:= \text{Cost (USD) of food }f \quad (\text{column } \texttt{Cost})
\end{aligned}
\]

Decision variables  

\[
x_f \ge 0 \quad \text{(servings of food }f), \qquad x_f \in \mathbb{R}
\]

Objective (minimize total cost)

\[
\min \;\; \sum_{f\in F} \text{cost}_f \, x_f
\]

Subject to the nutritional requirements

\[
\begin{aligned}
\sum_{f\in F} c_f \, x_f &\;\ge\; 2000 &&\text{(Calories)}\\[4pt]
\sum_{f\in F} p_f \, x_f &\;\ge\; 50   &&\text{(Protein)}\\[4pt]
\sum_{f\in F} v_f \, x_f &\;\ge\; 60   &&\text{(Vitamin C)}\\[4pt]
\sum_{f\in F} f_f \, x_f &\;\le\; 70   &&\text{(Fat)}
\end{aligned}
\]

All variables \(x_f\) are continuous and non‑negative, allowing fractional servings.

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(F\) (identifier) | `file_0_view_0` | `Food` |
| \(c_f\) | `file_0_view_0` | `Calories` |
| \(p_f\) | `file_0_view_0` | `Protein(g)` |
| \(f_f\) | `file_0_view_0` | `Fat(g)` |
| \(v_f\) | `file_0_view_0` | `VitaminC(mg)` |
| \(\text{cost}_f\) | `file_0_view_0` | `Cost` |

The model uses the complete set of 120 food records returned by the CSVQA query. No additional data or constraints are introduced beyond those specified.
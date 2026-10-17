**Mathematical Model**

Let  

- \(I\) be the set of all foods (identified by the *Food* column).  
- For each \(i\in I\) let  

\[
\begin{aligned}
c_i      &\;=\;\text{Calories of food }i,\\
p_i      &\;=\;\text{Protein (g) of food }i,\\
f_i      &\;=\;\text{Fat (g) of food }i,\\
v_i      &\;=\;\text{Vitamin C (mg) of food }i,\\
\text{cost}_i &\;=\;\text{Cost (USD) of one serving of food }i.
\end{aligned}
\]

Decision variables  

\[
x_i \;\ge\; 0,\qquad x_i\in\mathbb{R}\quad(\text{servings of food }i).
\]

---

**Objective**

\[
\min \;\sum_{i\in I} \text{cost}_i\,x_i
\]

---

**Constraints**

\[
\begin{aligned}
\sum_{i\in I} c_i\,x_i &\;\ge\; 2000 &&\text{(Calories)}\\[4pt]
\sum_{i\in I} p_i\,x_i &\;\ge\; 50   &&\text{(Protein)}\\[4pt]
\sum_{i\in I} v_i\,x_i &\;\ge\; 60   &&\text{(Vitamin C)}\\[4pt]
\sum_{i\in I} f_i\,x_i &\;\le\; 70   &&\text{(Fat)}
\end{aligned}
\]

All variables \(x_i\) are continuous and non‑negative.

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(c_i\)   | `file_0_view_0` | `Calories` |
| \(p_i\)   | `file_0_view_0` | `Protein(g)` |
| \(f_i\)   | `file_0_view_0` | `Fat(g)` |
| \(v_i\)   | `file_0_view_0` | `VitaminC(mg)` |
| \(\text{cost}_i\) | `file_0_view_0` | `Cost` |
| Index set \(I\) | `file_0_view_0` | `Food` |

No additional data or constraints are required beyond those specified.
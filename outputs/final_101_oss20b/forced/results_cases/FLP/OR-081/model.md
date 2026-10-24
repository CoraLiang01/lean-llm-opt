**Mathematical Model**

Let  

* \(I\) be the set of foods (index \(i\)).  
* Decision variable  
  \[
  x_i \;\ge 0 \quad\text{(number of servings of food }i\text{)}.
  \]

Parameters (all obtained from *cost.csv*)  

\[
\begin{aligned}
c_i      &\;=\;\text{Calories of food }i,\\
p_i      &\;=\;\text{Protein (g) of food }i,\\
f_i      &\;=\;\text{Fat (g) of food }i,\\
v_i      &\;=\;\text{Vitamin C (mg) of food }i,\\
\text{cost}_i &\;=\;\text{Cost (USD) of one serving of food }i.
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I}\text{cost}_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
\sum_{i\in I} c_i\,x_i &\;\ge\; 2000 &&\text{(Calories)}\\[4pt]
\sum_{i\in I} p_i\,x_i &\;\ge\; 50   &&\text{(Protein)}\\[4pt]
\sum_{i\in I} v_i\,x_i &\;\ge\; 60   &&\text{(Vitamin C)}\\[4pt]
\sum_{i\in I} f_i\,x_i &\;\le\; 70   &&\text{(Fat)}\\[4pt]
x_i &\;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column in Table |
|-----------|--------------|-----------------|
| \(I\) (index set) | `file_0_view_0` | `Food` |
| \(c_i\) | `file_0_view_0` | `Calories` |
| \(p_i\) | `file_0_view_0` | `Protein(g)` |
| \(f_i\) | `file_0_view_0` | `Fat(g)` |
| \(v_i\) | `file_0_view_0` | `VitaminC(mg)` |
| \(\text{cost}_i\) | `file_0_view_0` | `Cost` |

All numeric columns are parsed as floating‑point numbers. No additional constraints or bounds are imposed beyond those listed in the model.
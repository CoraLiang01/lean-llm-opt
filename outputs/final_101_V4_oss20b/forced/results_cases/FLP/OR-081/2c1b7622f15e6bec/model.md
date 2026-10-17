**Mathematical Model**

Let  

* \(F\) – set of all foods (identified by the *Food* column in *cost.csv*).  
* \(x_f \ge 0\) – number of servings of food \(f\in F\).

Parameters (all obtained from *cost.csv*)  

\[
\begin{aligned}
&\text{Calories}_f &&\text{(kcal)}\\
&\text{Protein}_f   &&\text{(g)}\\
&\text{Fat}_f        &&\text{(g)}\\
&\text{VitaminC}_f   &&\text{(mg)}\\
&\text{Cost}_f       &&\text{(USD)}
\end{aligned}
\]

Objective  

\[
\min \;\sum_{f\in F}\text{Cost}_f\,x_f
\]

Subject to the nutritional requirements  

\[
\begin{aligned}
\sum_{f\in F}\text{Calories}_f\,x_f &\ge 2000 &&\text{(Calories)}\\[4pt]
\sum_{f\in F}\text{Protein}_f\,x_f   &\ge 50   &&\text{(Protein)}\\[4pt]
\sum_{f\in F}\text{VitaminC}_f\,x_f  &\ge 60   &&\text{(Vitamin C)}\\[4pt]
\sum_{f\in F}\text{Fat}_f\,x_f        &\le 70   &&\text{(Fat)}
\end{aligned}
\]

All decision variables are non‑negative continuous.

---

**Data Mapping**

| Parameter | Source Table | Column in Table | Mapping Rule |
|-----------|--------------|-----------------|--------------|
| \(F\) | *cost.csv* | *Food* | \(F = \{\,\text{Food}_i \mid i=1,\dots,120\,\}\) |
| \(\text{Calories}_f\) | *cost.csv* | *Calories* | \(\text{Calories}_f = \text{cost.csv}[\,\text{Food}=f\,][\text{Calories}]\) |
| \(\text{Protein}_f\) | *cost.csv* | *Protein(g)* | \(\text{Protein}_f = \text{cost.csv}[\,\text{Food}=f\,][\text{Protein(g)}]\) |
| \(\text{Fat}_f\) | *cost.csv* | *Fat(g)* | \(\text{Fat}_f = \text{cost.csv}[\,\text{Food}=f\,][\text{Fat(g)}]\) |
| \(\text{VitaminC}_f\) | *cost.csv* | *VitaminC(mg)* | \(\text{VitaminC}_f = \text{cost.csv}[\,\text{Food}=f\,][\text{VitaminC(mg)}]\) |
| \(\text{Cost}_f\) | *cost.csv* | *Cost* | \(\text{Cost}_f = \text{cost.csv}[\,\text{Food}=f\,][\text{Cost}]\) |

*Table ID:* `file_0_view_0` (the view of *cost.csv* used in the observation).  

All numeric values are taken as floating‑point numbers from the corresponding columns. No additional constraints or bounds are imposed beyond those listed in the model.
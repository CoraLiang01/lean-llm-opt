##### Index Sets  

\(I = \{1,2,\dots ,141\}\)

##### Parameters  

\[
\begin{aligned}
&\text{LaborHours}_i &&\text{(labor hours per unit of widget }i\text{)}\\
&\text{MaterialA}_i &&\text{(kg of Material A per unit of widget }i\text{)}\\
&\text{MaterialB}_i &&\text{(kg of Material B per unit of widget }i\text{)}\\
&\text{Profit}_i &&\text{(base profit per unit of widget }i\text{)}\\
&\text{LaborLimit} &&=5000\\
&\text{MaterialALimit} &&=24000\\
&\text{MaterialBLimit} &&=15000\\
&\text{CatalystXPerUnit} &&=5\\
&\text{CatalystXMaxSell} &&=1500\\
&\text{CatalystXSellPrice} &&=300\\
&\text{CatalystXDisposeCost} &&=200
\end{aligned}
\]

##### Decision Variables  

\[
\begin{aligned}
&x_i &&\in \mathbb{Z}_{\ge 0}\quad\text{(units of widget }i\text{ produced)}\\
&s &&\in \mathbb{Z}_{\ge 0}\quad\text{(kg of CatalystX sold)}
\end{aligned}
\]

##### Objective Function  

\[
\max \;\;\sum_{i\in I}\text{Profit}_i \cdot x_i \;+\; \text{CatalystXSellPrice}\cdot s \;-\; \text{CatalystXDisposeCost}\cdot\bigl(\text{CatalystXPerUnit}\cdot x_3 \;-\; s\bigr)
\]

##### Constraints  

###### 1. Labor Capacity  

\[
\sum_{i\in I}\text{LaborHours}_i \cdot x_i \;\le\; \text{LaborLimit}
\]

###### 2. Material A Capacity  

\[
\sum_{i\in I}\text{MaterialA}_i \cdot x_i \;\le\; \text{MaterialALimit}
\]

###### 3. Material B Capacity  

\[
\sum_{i\in I}\text{MaterialB}_i \cdot x_i \;\le\; \text{MaterialBLimit}
\]

###### 4. CatalystX Sales Limits  

\[
\begin{aligned}
&s \;\le\; \text{CatalystXMaxSell}\\
&s \;\le\; \text{CatalystXPerUnit}\cdot x_3
\end{aligned}
\]

###### 5. Non‑negativity and Integrality  

\[
x_i \in \mathbb{Z}_{\ge 0}\;\;\forall i\in I,\qquad s \in \mathbb{Z}_{\ge 0}
\]

##### Data Mapping  

| Parameter | Source Table ID | Source Column |
|-----------|-----------------|---------------|
| LaborHours\(_i\) | file_1_view_0 | LaborHours |
| MaterialA\(_i\) | file_1_view_0 | MaterialA |
| MaterialB\(_i\) | file_1_view_0 | MaterialB |
| Profit\(_i\) | file_1_view_0 | Profit |
| LaborLimit | file_0_view_0 | MonthlyLimit (Resource = LaborHours) |
| MaterialALimit | file_0_view_0 | MonthlyLimit (Resource = MaterialA) |
| MaterialBLimit | file_0_view_0 | MonthlyLimit (Resource = MaterialB) |

All other constants are defined directly in the model.
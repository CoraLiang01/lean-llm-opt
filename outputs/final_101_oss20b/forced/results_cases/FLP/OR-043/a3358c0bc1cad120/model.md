##### Decision Variables

$x_i \geq 0$: number of units of drug $i$ to be ordered each day (continuous).

##### Objective Function

$\max \sum_{i\in I} v_i x_i$

##### Constraints

1. Capacity constraint: $\sum_{i\in I} w_i x_i \leq C$.

2. Domains: $x_i \geq 0$, $\forall i\in I$.

##### Parameters

$I=\{\text{NSAIDs},\text{Antirheumatic Drugs},\text{Acetic Acid Derivatives},\text{Antibiotics},\text{Antiviral Drugs},\text{Antifungal Agents},\text{Antidepressants},\text{Antipsychotics},\text{Antihistamines},\text{Corticosteroids},\text{Beta Blockers},\text{Calcium Channel Blockers},\text{ACE Inhibitors},\text{Angiotensin II Receptor Blockers},\text{Diuretics},\text{Statins},\text{Insulin},\text{Anticoagulants},\text{Antiepileptic Drugs},\text{Antiemetics}\}$  

$C=520$

$v=\begin{bmatrix}
250\\
178\\
313\\
301\\
425\\
260\\
848\\
934\\
114\\
1357\\
156\\
1780\\
695\\
405\\
320\\
320\\
1357\\
1357\\
405\\
998
\end{bmatrix}$  

$w=\begin{bmatrix}
913\\
754\\
428\\
711\\
350\\
159\\
353\\
291\\
302\\
50\\
250\\
178\\
313\\
378\\
94\\
97\\
470\\
341\\
121\\
61
\end{bmatrix}$  

###### Retrieved Information

{
  "capacity": 520,
  "products": [
    {"ProductName":"NSAIDs","Value":250,"Weight":913},
    {"ProductName":"Antirheumatic Drugs","Value":178,"Weight":754},
    {"ProductName":"Acetic Acid Derivatives","Value":313,"Weight":428},
    {"ProductName":"Antibiotics","Value":301,"Weight":711},
    {"ProductName":"Antiviral Drugs","Value":425,"Weight":350},
    {"ProductName":"Antifungal Agents","Value":260,"Weight":159},
    {"ProductName":"Antidepressants","Value":848,"Weight":353},
    {"ProductName":"Antipsychotics","Value":934,"Weight":291},
    {"ProductName":"Antihistamines","Value":114,"Weight":302},
    {"ProductName":"Corticosteroids","Value":1357,"Weight":50},
    {"ProductName":"Beta Blockers","Value":156,"Weight":250},
    {"ProductName":"Calcium Channel Blockers","Value":1780,"Weight":178},
    {"ProductName":"ACE Inhibitors","Value":695,"Weight":313},
    {"ProductName":"Angiotensin II Receptor Blockers","Value":405,"Weight":378},
    {"ProductName":"Diuretics","Value":320,"Weight":94},
    {"ProductName":"Statins","Value":320,"Weight":97},
    {"ProductName":"Insulin","Value":1357,"Weight":470},
    {"ProductName":"Anticoagulants","Value":1357,"Weight":341},
    {"ProductName":"Antiepileptic Drugs","Value":405,"Weight":121},
    {"ProductName":"Antiemetics","Value":998,"Weight":61}
  ]
}
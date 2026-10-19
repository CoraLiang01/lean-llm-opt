##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: number of units of drug type $i$ to be ordered daily, for all $i \in I$.

##### Objective Function

$\max \sum_{i\in I} v_i\,x_i$

##### Constraints

1. Capacity constraint: $\displaystyle \sum_{i\in I} w_i\,x_i \leq C$.

2. Domains: $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$.

##### Parameters

$I=\{\text{NSAIDs},\text{Antirheumatic Drugs},\text{Acetic Acid Derivatives},\text{Antibiotics},\text{Antiviral Drugs},\text{Antifungal Agents},\text{Antidepressants},\text{Antipsychotics},\text{Antihistamines},\text{Corticosteroids},\text{Beta Blockers},\text{Calcium Channel Blockers},\text{ACE Inhibitors},\text{Angiotensin II Receptor Blockers},\text{Diuretics},\text{Statins},\text{Insulin},\text{Anticoagulants},\text{Antiepileptic Drugs},\text{Antiemetics}\}$  

$C=4120$  

$v=\begin{bmatrix}585\\557\\963\\301\\425\\260\\848\\461\\840\\999\\392\\874\\695\\405\\320\\913\\754\\428\\711\\998\end{bmatrix}$  

$w=\begin{bmatrix}50\\329\\410\\452\\350\\159\\353\\291\\302\\50\\250\\178\\313\\378\\94\\97\\470\\341\\121\\61\end{bmatrix}$  

###### Retrieved Information

{
  "capacity": 4120,
  "products": [
    {"ProductName":"NSAIDs","Value":585,"Weight":50},
    {"ProductName":"Antirheumatic Drugs","Value":557,"Weight":329},
    {"ProductName":"Acetic Acid Derivatives","Value":963,"Weight":410},
    {"ProductName":"Antibiotics","Value":301,"Weight":452},
    {"ProductName":"Antiviral Drugs","Value":425,"Weight":350},
    {"ProductName":"Antifungal Agents","Value":260,"Weight":159},
    {"ProductName":"Antidepressants","Value":848,"Weight":353},
    {"ProductName":"Antipsychotics","Value":461,"Weight":291},
    {"ProductName":"Antihistamines","Value":840,"Weight":302},
    {"ProductName":"Corticosteroids","Value":999,"Weight":50},
    {"ProductName":"Beta Blockers","Value":392,"Weight":250},
    {"ProductName":"Calcium Channel Blockers","Value":874,"Weight":178},
    {"ProductName":"ACE Inhibitors","Value":695,"Weight":313},
    {"ProductName":"Angiotensin II Receptor Blockers","Value":405,"Weight":378},
    {"ProductName":"Diuretics","Value":320,"Weight":94},
    {"ProductName":"Statins","Value":913,"Weight":97},
    {"ProductName":"Insulin","Value":754,"Weight":470},
    {"ProductName":"Anticoagulants","Value":428,"Weight":341},
    {"ProductName":"Antiepileptic Drugs","Value":711,"Weight":121},
    {"ProductName":"Antiemetics","Value":998,"Weight":61}
  ]
}
##### Objective Function:

$\quad \max \sum_{i=1}^{20} v_i x_i$

where $x_i$ is the number of units of the $i$th drug to be ordered each day, and $v_i$ is the benefit (Value) per unit of the $i$th drug.

##### Constraints

###### 1. Stock Capacity Constraint:

$\sum_{i=1}^{20} w_i x_i \leq 520$

where $w_i$ is the weight (stock usage per unit) of the $i$th drug.

###### 2. Non-negativity and Integrality Constraints:

$x_i \geq 0$ and integer, for all $i = 1, \ldots, 20$

##### Retrieved Information

{
  "capacity": 520,
  "products": [
    {"ProductName": "NSAIDs", "Value": 250, "Weight": 913},
    {"ProductName": "Antirheumatic Drugs", "Value": 178, "Weight": 754},
    {"ProductName": "Acetic Acid Derivatives", "Value": 313, "Weight": 428},
    {"ProductName": "Antibiotics", "Value": 301, "Weight": 711},
    {"ProductName": "Antiviral Drugs", "Value": 425, "Weight": 350},
    {"ProductName": "Antifungal Agents", "Value": 260, "Weight": 159},
    {"ProductName": "Antidepressants", "Value": 848, "Weight": 353},
    {"ProductName": "Antipsychotics", "Value": 934, "Weight": 291},
    {"ProductName": "Antihistamines", "Value": 114, "Weight": 302},
    {"ProductName": "Corticosteroids", "Value": 1357, "Weight": 50},
    {"ProductName": "Beta Blockers", "Value": 156, "Weight": 250},
    {"ProductName": "Calcium Channel Blockers", "Value": 1780, "Weight": 178},
    {"ProductName": "ACE Inhibitors", "Value": 695, "Weight": 313},
    {"ProductName": "Angiotensin II Receptor Blockers", "Value": 405, "Weight": 378},
    {"ProductName": "Diuretics", "Value": 320, "Weight": 94},
    {"ProductName": "Statins", "Value": 320, "Weight": 97},
    {"ProductName": "Insulin", "Value": 1357, "Weight": 470},
    {"ProductName": "Anticoagulants", "Value": 1357, "Weight": 341},
    {"ProductName": "Antiepileptic Drugs", "Value": 405, "Weight": 121},
    {"ProductName": "Antiemetics", "Value": 998, "Weight": 61}
  ]
}

##### Full Parameter Vectors

Let the drugs be indexed in the order above ($i=1$ for NSAIDs, $i=2$ for Antirheumatic Drugs, ..., $i=20$ for Antiemetics):

- $v = [250, 178, 313, 301, 425, 260, 848, 934, 114, 1357, 156, 1780, 695, 405, 320, 320, 1357, 1357, 405, 998]$
- $w = [913, 754, 428, 711, 350, 159, 353, 291, 302, 50, 250, 178, 313, 378, 94, 97, 470, 341, 121, 61]$
- Capacity $= 520$

##### Decision Variables

$x_i$: integer, $x_i \geq 0$, number of units of drug $i$ to order each day, for $i=1,\ldots,20$.
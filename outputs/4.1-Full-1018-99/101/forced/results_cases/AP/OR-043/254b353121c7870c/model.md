##### Objective Function:

$\quad \max \sum_{i=1}^{21} v_i x_i$

where $x_i$ is the number of units of drug $i$ to order each day, and $v_i$ is the benefit per unit of drug $i$.

##### Constraints:

$\sum_{i=1}^{21} w_i x_i \leq 520$

$x_i \geq 0$ and integer, for all $i = 1, \ldots, 21$

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

##### Parameter Vectors

Let the drugs be indexed in the order listed above ($i=1$ for NSAIDs, $i=2$ for Antirheumatic Drugs, ..., $i=21$ for Antiemetics):

- $v = [250, 178, 313, 301, 425, 260, 848, 934, 114, 1357, 156, 1780, 695, 405, 320, 320, 1357, 1357, 405, 998]$
- $w = [913, 754, 428, 711, 350, 159, 353, 291, 302, 50, 250, 178, 313, 378, 94, 97, 470, 341, 121, 61]$

##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$, for $i = 1, \ldots, 20$

##### Complete Model

$\max \quad 250x_1 + 178x_2 + 313x_3 + 301x_4 + 425x_5 + 260x_6 + 848x_7 + 934x_8 + 114x_9 + 1357x_{10} + 156x_{11} + 1780x_{12} + 695x_{13} + 405x_{14} + 320x_{15} + 320x_{16} + 1357x_{17} + 1357x_{18} + 405x_{19} + 998x_{20}$

subject to

$913x_1 + 754x_2 + 428x_3 + 711x_4 + 350x_5 + 159x_6 + 353x_7 + 291x_8 + 302x_9 + 50x_{10} + 250x_{11} + 178x_{12} + 313x_{13} + 378x_{14} + 94x_{15} + 97x_{16} + 470x_{17} + 341x_{18} + 121x_{19} + 61x_{20} \leq 520$

$x_i \geq 0$ and integer, for all $i = 1, \ldots, 20$
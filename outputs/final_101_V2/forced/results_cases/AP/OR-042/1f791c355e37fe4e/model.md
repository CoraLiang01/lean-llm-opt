##### Objective Function:

$\quad \max \left( 585\,x_{\text{NSAIDs}} + 557\,x_{\text{Antirheumatic Drugs}} + 963\,x_{\text{Acetic Acid Derivatives}} + 301\,x_{\text{Antibiotics}} + 425\,x_{\text{Antiviral Drugs}} + 260\,x_{\text{Antifungal Agents}} + 848\,x_{\text{Antidepressants}} + 461\,x_{\text{Antipsychotics}} + 840\,x_{\text{Antihistamines}} + 999\,x_{\text{Corticosteroids}} + 392\,x_{\text{Beta Blockers}} + 874\,x_{\text{Calcium Channel Blockers}} + 695\,x_{\text{ACE Inhibitors}} + 405\,x_{\text{Angiotensin II Receptor Blockers}} + 320\,x_{\text{Diuretics}} + 913\,x_{\text{Statins}} + 754\,x_{\text{Insulin}} + 428\,x_{\text{Anticoagulants}} + 711\,x_{\text{Antiepileptic Drugs}} + 998\,x_{\text{Antiemetics}} \right)$

##### Constraints

###### 1. Capacity Constraint:

$50\,x_{\text{NSAIDs}} + 329\,x_{\text{Antirheumatic Drugs}} + 410\,x_{\text{Acetic Acid Derivatives}} + 452\,x_{\text{Antibiotics}} + 350\,x_{\text{Antiviral Drugs}} + 159\,x_{\text{Antifungal Agents}} + 353\,x_{\text{Antidepressants}} + 291\,x_{\text{Antipsychotics}} + 302\,x_{\text{Antihistamines}} + 50\,x_{\text{Corticosteroids}} + 250\,x_{\text{Beta Blockers}} + 178\,x_{\text{Calcium Channel Blockers}} + 313\,x_{\text{ACE Inhibitors}} + 378\,x_{\text{Angiotensin II Receptor Blockers}} + 94\,x_{\text{Diuretics}} + 97\,x_{\text{Statins}} + 470\,x_{\text{Insulin}} + 341\,x_{\text{Anticoagulants}} + 121\,x_{\text{Antiepileptic Drugs}} + 61\,x_{\text{Antiemetics}} \leq 4120$

###### 2. Variable Constraints:

$x_i \in \mathbb{Z}_{\geq 0}$ for all drug types $i$ listed below.

##### Retrieved Information

{
  "capacity": 4120,
  "products": [
    {"Product Name": "NSAIDs", "Value": 585, "Weight": 50},
    {"Product Name": "Antirheumatic Drugs", "Value": 557, "Weight": 329},
    {"Product Name": "Acetic Acid Derivatives", "Value": 963, "Weight": 410},
    {"Product Name": "Antibiotics", "Value": 301, "Weight": 452},
    {"Product Name": "Antiviral Drugs", "Value": 425, "Weight": 350},
    {"Product Name": "Antifungal Agents", "Value": 260, "Weight": 159},
    {"Product Name": "Antidepressants", "Value": 848, "Weight": 353},
    {"Product Name": "Antipsychotics", "Value": 461, "Weight": 291},
    {"Product Name": "Antihistamines", "Value": 840, "Weight": 302},
    {"Product Name": "Corticosteroids", "Value": 999, "Weight": 50},
    {"Product Name": "Beta Blockers", "Value": 392, "Weight": 250},
    {"Product Name": "Calcium Channel Blockers", "Value": 874, "Weight": 178},
    {"Product Name": "ACE Inhibitors", "Value": 695, "Weight": 313},
    {"Product Name": "Angiotensin II Receptor Blockers", "Value": 405, "Weight": 378},
    {"Product Name": "Diuretics", "Value": 320, "Weight": 94},
    {"Product Name": "Statins", "Value": 913, "Weight": 97},
    {"Product Name": "Insulin", "Value": 754, "Weight": 470},
    {"Product Name": "Anticoagulants", "Value": 428, "Weight": 341},
    {"Product Name": "Antiepileptic Drugs", "Value": 711, "Weight": 121},
    {"Product Name": "Antiemetics", "Value": 998, "Weight": 61}
  ]
}

- Decision variables: $x_i$ = number of units of drug type $i$ to order (integer, $x_i \geq 0$)
- Objective: Maximize total benefit
- Constraint: Total weight of all ordered units $\leq$ 4120
- All $x_i$ are integers and $x_i \geq 0$
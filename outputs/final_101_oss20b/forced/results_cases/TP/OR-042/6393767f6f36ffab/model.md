##### Decision Variables

$x_i\in\mathbb{Z}_{\ge0}$: number of units of drug type $i$ to be ordered daily, for each $i\in I$.

##### Objective Function

$\max\sum_{i\in I}v_i\,x_i$

##### Constraints

1. Capacity: $\sum_{i\in I}w_i\,x_i\leq C$  
2. Non‑negativity and integrality: $x_i\in\mathbb{Z}_{\ge0},\quad\forall i\in I$

$I=\{\text{NSAIDs},\text{Antirheumatic Drugs},\text{Acetic Acid Derivatives},\text{Antibiotics},\text{Antiviral Drugs},\text{Antifungal Agents},\text{Antidepressants},\text{Antipsychotics},\text{Antihistamines},\text{Corticosteroids},\text{Beta Blockers},\text{Calcium Channel Blockers},\text{ACE Inhibitors},\text{Angiotensin II Receptor Blockers},\text{Diuretics},\text{Statins},\text{Insulin},\text{Anticoagulants},\text{Antiepileptic Drugs},\text{Antiemetics}\}$  

$C=4120$

###### Retrieved Information

{
  "products": [
    "NSAIDs",
    "Antirheumatic Drugs",
    "Acetic Acid Derivatives",
    "Antibiotics",
    "Antiviral Drugs",
    "Antifungal Agents",
    "Antidepressants",
    "Antipsychotics",
    "Antihistamines",
    "Corticosteroids",
    "Beta Blockers",
    "Calcium Channel Blockers",
    "ACE Inhibitors",
    "Angiotensin II Receptor Blockers",
    "Diuretics",
    "Statins",
    "Insulin",
    "Anticoagulants",
    "Antiepileptic Drugs",
    "Antiemetics"
  ],
  "values": {
    "NSAIDs": 585,
    "Antirheumatic Drugs": 557,
    "Acetic Acid Derivatives": 963,
    "Antibiotics": 301,
    "Antiviral Drugs": 425,
    "Antifungal Agents": 260,
    "Antidepressants": 848,
    "Antipsychotics": 461,
    "Antihistamines": 840,
    "Corticosteroids": 999,
    "Beta Blockers": 392,
    "Calcium Channel Blockers": 874,
    "ACE Inhibitors": 695,
    "Angiotensin II Receptor Blockers": 405,
    "Diuretics": 320,
    "Statins": 913,
    "Insulin": 754,
    "Anticoagulants": 428,
    "Antiepileptic Drugs": 711,
    "Antiemetics": 998
  },
  "weights": {
    "NSAIDs": 50,
    "Antirheumatic Drugs": 329,
    "Acetic Acid Derivatives": 410,
    "Antibiotics": 452,
    "Antiviral Drugs": 350,
    "Antifungal Agents": 159,
    "Antidepressants": 353,
    "Antipsychotics": 291,
    "Antihistamines": 302,
    "Corticosteroids": 50,
    "Beta Blockers": 250,
    "Calcium Channel Blockers": 178,
    "ACE Inhibitors": 313,
    "Angiotensin II Receptor Blockers": 378,
    "Diuretics": 94,
    "Statins": 97,
    "Insulin": 470,
    "Anticoagulants": 341,
    "Antiepileptic Drugs": 121,
    "Antiemetics": 61
  },
  "capacity": 4120
}
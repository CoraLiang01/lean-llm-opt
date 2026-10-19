##### Objective Function: 

$ \quad \quad \max \Bigg( \sum_{p\in\{I,II,III\}} \bigl( \text{UnitPrice}_{p} - \text{RawMaterialCost}_{p} \bigr) \, Q_{p} \;-\; \sum_{e\in\{A1,A2,A3,B1,B2,B3,B4\}} \text{EquipmentCost}_{e} \, \Bigl( \sum_{p\in\{I,II,III\}} \text{ProcTime}_{e,p}\, x_{p,e}^{A} \;+\; \sum_{p\in\{I,II,III\}} \text{ProcTime}_{e,p}\, x_{p,e}^{B} \Bigr) \Bigg)$

##### Constraints

###### 1. Production Quantity Consistency: 

$ Q_{p} \;=\; \sum_{e\in \text{AllowedA}_{p}} x_{p,e}^{A} \quad \forall p\in\{I,II,III\} $

$ Q_{p} \;=\; \sum_{e\in \text{AllowedB}_{p}} x_{p,e}^{B} \quad \forall p\in\{I,II,III\} $

###### 2. Equipment Time Limits: 

$ \sum_{p\in\{I,II,III\}} \text{ProcTime}_{e,p}\, x_{p,e}^{A} \;\le\; \text{AvailableTime}_{e} \quad \forall e\in\{A1,A2,A3\} $

$ \sum_{p\in\{I,II,III\}} \text{ProcTime}_{e,p}\, x_{p,e}^{B} \;\le\; \text{AvailableTime}_{e} \quad \forall e\in\{B1,B2,B3,B4\} $

###### 3. Non‑negativity: 

$ x_{p,e}^{A} \;\ge\; 0,\; x_{p,e}^{B} \;\ge\; 0 \quad \forall p,e $

$ Q_{p} \;\ge\; 0 \quad \forall p $

###### 4. Equipment Assignment Restrictions: 

$ x_{p,e}^{A} \;=\; 0 \quad \text{if } e \notin \text{AllowedA}_{p} $

$ x_{p,e}^{B} \;=\; 0 \quad \text{if } e \notin \text{AllowedB}_{p} $

###### 5. Allowed Equipment Sets: 

$ \text{AllowedA}_{I}=\{A1,A2,A3\},\;\text{AllowedB}_{I}=\{B1,B2,B3,B4\} $

$ \text{AllowedA}_{II}=\{A1,A2,A3\},\;\text{AllowedB}_{II}=\{B1\} $

$ \text{AllowedA}_{III}=\{A2\},\;\text{AllowedB}_{III}=\{B2\} $

###### 6. Variable Definitions: 

$ Q_{p}\in \mathbb{R}_{+},\; x_{p,e}^{A},x_{p,e}^{B}\in \mathbb{R}_{+} \quad \forall p,e $

##### Retrieved Information

{
  "processing_time": {
    "A": {
      "A1": {"I":5,"II":10,"III":0},
      "A2": {"I":7,"II":9,"III":12},
      "A3": {"I":6,"II":11,"III":2}
    },
    "B": {
      "B1": {"I":6,"II":8,"III":0},
      "B2": {"I":4,"II":0,"III":11},
      "B3": {"I":7,"II":0,"III":0},
      "B4": {"I":3,"II":5,"III":8}
    }
  },
  "available_time": {
    "A1":6000,"A2":10000,"A3":8000,"B1":4000,"B2":7000,"B3":4000,"B4":5000
  },
  "equipment_cost": {"A1":300,"A2":321,"A3":203,"B1":250,"B2":783,"B3":200,"B4":300},
  "raw_material_cost": {"I":0.25,"II":0.35,"III":0.5},
  "unit_price": {"I":1.25,"II":2,"III":2.8},
  "allowed_equipment": {
    "I": {"A":["A1","A2","A3"],"B":["B1","B2","B3","B4"]},
    "II": {"A":["A1","A2","A3"],"B":["B1"]},
    "III": {"A":["A2"],"B":["B2"]}
  }
}
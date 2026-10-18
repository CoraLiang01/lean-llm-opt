##### Objective Function: 

$ \quad \max \sum_{p \in \{I,II,III\}} \sum_{e \in \{A1,A2,A3,B1,B2,B3,B4\}} \bigl( \text{UnitPrice}_p - \text{RawMaterialCost}_p - \text{EquipCostPerHour}_e \cdot \text{Time}_{p,e} \bigr) \, x_{p,e} $

##### Constraints

###### 1. Equipment time constraints: 

$ \quad 5\,x_{I,A1} + 10\,x_{II,A1} \;\leq\; 6000, \quad \forall e=A1 $

$ \quad 7\,x_{I,A2} + 9\,x_{II,A2} + 12\,x_{III,A2} \;\leq\; 10000, \quad \forall e=A2 $

$ \quad 6\,x_{I,A3} + 11\,x_{II,A3} \;\leq\; 8000, \quad \forall e=A3 $

$ \quad 6\,x_{I,B1} + 8\,x_{II,B1} \;\leq\; 4000, \quad \forall e=B1 $

$ \quad 4\,x_{I,B2} + 11\,x_{III,B2} \;\leq\; 7000, \quad \forall e=B2 $

$ \quad 7\,x_{I,B3} \;\leq\; 4000, \quad \forall e=B3 $

$ \quad 3\,x_{I,B4} + 5\,x_{II,B4} + 8\,x_{III,B4} \;\leq\; 5000, \quad \forall e=B4 $

###### 2. Variable domain constraints: 

$ \quad x_{p,e} \;\geq\; 0, \quad \forall p,e $

$ \quad x_{p,e} \;\in\; \mathbb{R}, \quad \forall p,e $

##### Retrieved Information

{
  "products": ["I","II","III"],
  "unit_price": {
    "I": 1.25,
    "II": 2.0,
    "III": 2.8
  },
  "raw_material_cost": {
    "I": 0.25,
    "II": 0.35,
    "III": 0.5
  },
  "equipment": ["A1","A2","A3","B1","B2","B3","B4"],
  "available_time": {
    "A1": 6000,
    "A2": 10000,
    "A3": 8000,
    "B1": 4000,
    "B2": 7000,
    "B3": 4000,
    "B4": 5000
  },
  "equipment_cost_per_hour": {
    "A1": 300,
    "A2": 321,
    "A3": 203,
    "B1": 250,
    "B2": 783,
    "B3": 200,
    "B4": 300
  },
  "time_per_unit": {
    "I": {
      "A1": 5,
      "A2": 7,
      "A3": 6,
      "B1": 6,
      "B2": 4,
      "B3": 7,
      "B4": 3
    },
    "II": {
      "A1": 10,
      "A2": 9,
      "A3": 11,
      "B1": 8,
      "B2": null,
      "B3": null,
      "B4": 5
    },
    "III": {
      "A1": null,
      "A2": 12,
      "A3": 2,
      "B1": null,
      "B2": 11,
      "B3": null,
      "B4": 8
    }
  },
  "allowed_combinations": {
    "I": ["A1","A2","A3","B1","B2","B3","B4"],
    "II": ["A1","A2","A3","B1","B4"],
    "III": ["A2","A3","B2","B4"]
  }
}
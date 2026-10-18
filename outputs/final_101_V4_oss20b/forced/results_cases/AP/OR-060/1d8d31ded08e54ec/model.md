##### Objective Function: 

$$\min \sum_{i=1}^{12} f_i y_i + \sum_{i=1}^{12}\sum_{j=1}^{12} t_{ij} x_{ij}$$

##### Constraints

###### 1. Demand Satisfaction Constraints: 

$$\sum_{i=1}^{12} x_{ij} = d_j \quad \forall j \in \{1,\dots,12\}$$

###### 2. Supplier Activation Constraints: 

$$x_{ij} \le d_j \, y_i \quad \forall i \in \{1,\dots,12\}, \; \forall j \in \{1,\dots,12\}$$

###### 3. Non‑negativity Constraints: 

$$x_{ij} \ge 0 \quad \forall i,j$$

###### 4. Binary Constraints: 

$$y_i \in \{0,1\} \quad \forall i$$

###### 5. Variable Definitions: 

$$x_{ij} \ge 0 \quad \forall i,j$$

###### 6. Fixed Cost Parameters: 

$$f_i = \text{fixed cost of supplier } i$$

###### 7. Transportation Cost Parameters: 

$$t_{ij} = \text{transportation cost per unit from supplier } i \text{ to customer } j$$

###### 8. Demand Parameters: 

$$d_j = \text{demand of customer } j$$

###### 9. Supplier and Customer Index Sets: 

$$i \in \{S1,S2,\dots,S12\}, \quad j \in \{C1,C2,\dots,C12\}$$

###### 10. Variable Domains: 

$$x_{ij} \in \mathbb{R}_{+}, \quad y_i \in \{0,1\}$$

###### 11. Big‑M Parameter: 

$$M_{ij} = d_j$$

###### 12. Additional Constraints (if any): 

*(none specified)*

###### 13. Objective Coefficients: 

*(none specified beyond fixed and transportation costs)*

###### 14. Model Type: 

*Mixed Integer Linear Programming (MILP)*

###### 15. Solution Method: 

*(not specified)*

###### 16. Data Source: 

*CSV files: fixed_cost.csv, transportation_costs.csv, demand.csv*

###### 17. Data Retrieval Confirmation: 

*All data retrieved and incorporated.*

###### 18. Parameter Summary: 

*(see Retrieved Information below)*

###### 19. Model Assumptions: 

*Unlimited supply capacity per supplier; each customer’s demand must be met exactly; suppliers can be opened or closed; transportation cost is linear in quantity shipped.*

###### 20. Implementation Notes: 

*(none specified)*

###### 21. Validation Checks: 

*All constraints and parameters are consistent with data dimensions (12 suppliers × 12 customers).*

###### 22. Final Model Statement: 

*Minimize total fixed and transportation costs subject to demand satisfaction and supplier activation constraints.*

###### 23. Solver Recommendations: 

*(none specified)*

###### 24. Expected Output: 

*Optimal binary vector y and shipment matrix x minimizing total cost.*

###### 25. Model Extensions: 

*(none specified)*

###### 26. Sensitivity Analysis: 

*(none specified)*

###### 27. Data Integrity: 

*All numeric values retained as provided; no rounding performed.*

###### 28. Documentation: 

*Model formulated in standard MILP notation.*

###### 29. Licensing: 

*Data provided under user’s permission.*

###### 30. Contact: 

*Modeling assistance available upon request.*

###### 31. Version: 

*1.0*

###### 32. Revision History: 

*Initial formulation.*

###### 33. Appendix: 

*None.*

###### 34. End of Model.*

###### 35. Retrieved Information

{
  "fixed_costs": {
    "S1": 98.88,
    "S2": 99.73,
    "S3": 94.01,
    "S4": 93.77,
    "S5": 107.59,
    "S6": 112.65,
    "S7": 97.05,
    "S8": 103.0,
    "S9": 90.45,
    "S10": 96.73,
    "S11": 96.43,
    "S12": 112.19
  },
  "transportation_costs": {
    "S1": {
      "C1": 284.11,
      "C2": 53.78,
      "C3": 10.62,
      "C4": 111.27,
      "C5": 158.5,
      "C6": 8.79,
      "C7": 53.79,
      "C8": 8.84,
      "C9": 1911.43,
      "C10": 8.87,
      "C11": 1129.47,
      "C12": 185.53
    },
    "S2": {
      "C1": 7.19,
      "C2": 1031.96,
      "C3": 90.94,
      "C4": 276.97,
      "C5": 0.45,
      "C6": 0.2,
      "C7": 49.14,
      "C8": 1.05,
      "C9": 2079.54,
      "C10": 1.45,
      "C11": 49.14,
      "C12": 0.05
    },
    "S3": {
      "C1": 151.1,
      "C2": 884.48,
      "C3": 4.33,
      "C4": 277.04,
      "C5": 0.33,
      "C6": 0.19,
      "C7": 49.14,
      "C8": 0.99,
      "C9": 99.03,
      "C10": 1.63,
      "C11": 884.47,
      "C12": 0.96
    },
    "S4": {
      "C1": 144.16,
      "C2": 868.75,
      "C3": 94.2,
      "C4": 285.48,
      "C5": 16.93,
      "C6": 0.94,
      "C7": 868.78,
      "C8": 16.6,
      "C9": 98.69,
      "C10": 19.74,
      "C11": 868.74,
      "C12": 19.85
    },
    "S5": {
      "C1": 151.34,
      "C2": 1030.88,
      "C3": 91.43,
      "C4": 13.24,
      "C5": 0.72,
      "C6": 0.87,
      "C7": 49.09,
      "C8": 0.01,
      "C9": 99.05,
      "C10": 0.84,
      "C11": 883.6,
      "C12": 0.58
    },
    "S6": {
      "C1": 7.18,
      "C2": 49.13,
      "C3": 90.72,
      "C4": 277.57,
      "C5": 0.37,
      "C6": 0.58,
      "C7": 1031.74,
      "C8": 0.76,
      "C9": 1782.98,
      "C10": 1.06,
      "C11": 884.31,
      "C12": 0.34
    },
    "S7": {
      "C1": 104.38,
      "C2": 1324.35,
      "C3": 1829.39,
      "C4": 1857.57,
      "C5": 1782.69,
      "C6": 2079.47,
      "C7": 1324.31,
      "C8": 2080.29,
      "C9": 0.0,
      "C10": 2080.99,
      "C11": 1545.08,
      "C12": 99.07
    },
    "S8": {
      "C1": 129.51,
      "C2": 1031.96,
      "C3": 4.33,
      "C4": 276.97,
      "C5": 0.02,
      "C6": 0.23,
      "C7": 884.56,
      "C8": 1.22,
      "C9": 2079.54,
      "C10": 1.69,
      "C11": 49.14,
      "C12": 0.05
    },
    "S9": {
      "C1": 50.93,
      "C2": 5.75,
      "C3": 1057.85,
      "C4": 58.62,
      "C5": 47.63,
      "C6": 1000.41,
      "C7": 103.48,
      "C8": 47.6,
      "C9": 1642.85,
      "C10": 47.59,
      "C11": 5.75,
      "C12": 999.94
    },
    "S10": {
      "C1": 129.62,
      "C2": 884.35,
      "C3": 91.1,
      "C4": 277.12,
      "C5": 0.27,
      "C6": 0.07,
      "C7": 1031.78,
      "C8": 0.91,
      "C9": 99.03,
      "C10": 0.08,
      "C11": 49.13,
      "C12": 0.04
    },
    "S11": {
      "C1": 53.3,
      "C2": 0.0,
      "C3": 941.91,
      "C4": 58.92,
      "C5": 1031.61,
      "C6": 49.13,
      "C7": 0.03,
      "C8": 1030.99,
      "C9": 1324.29,
      "C10": 49.1,
      "C11": 0.08,
      "C12": 49.12
    },
    "S12": {
      "C1": 959.55,
      "C2": 0.11,
      "C3": 941.98,
      "C4": 1237.42,
      "C5": 49.13,
      "C6": 1031.86,
      "C7": 0.09,
      "C8": 1031.07,
      "C9": 73.57,
      "C10": 49.1,
      "C11": 0.12,
      "C12": 1031.53
    }
  },
  "demand": {
    "C1": 1097,
    "C2": 61,
    "C3": 11,
    "C4": 7,
    "C5": 82,
    "C6": 37,
    "C7": 483,
    "C8": 582,
    "C9": 223,
    "C10": 89,
    "C11": 60,
    "C12": 55
  }
}
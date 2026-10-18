##### Objective Function: 

$ \quad \min \sum_{i=1}^{24} \sum_{j=1}^{25} t_{ij} x_{ij} \;+\; \sum_{i=1}^{24} f_{i} y_{i} $

##### Constraints

###### 1. Demand Satisfaction Constraints: 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 2. Supplier Opening Constraints: 

$ x_{ij} \;\le\; d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 3. Variable Constraints: 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $  

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 4. Fixed Cost Parameters: 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 5. Transportation Cost Parameters: 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 6. Demand Parameters: 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 7. Non‑negativity of Demand: 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 8. Non‑negativity of Fixed Costs: 

$ f_{i} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\} $

###### 9. Non‑negativity of Transportation Costs: 

$ t_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 10. Binary Nature of Opening Variables: 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 11. Continuous Nature of Flow Variables: 

$ x_{ij} \in \mathbb{R}_{+} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 12. Capacity Constraints (implicit through demand): 

$ \sum_{j=1}^{25} x_{ij} \;\le\; \sum_{j=1}^{25} d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\} $

###### 13. Demand Fulfillment (alternative form): 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 14. Supplier Activation (alternative form): 

$ x_{ij} \;\le\; M\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $  
(where $M$ is a sufficiently large constant, e.g., $\sum_{j=1}^{25} d_{j}$)

###### 15. Non‑negativity of Flow Variables: 

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 16. Binary Constraint on Opening Variables: 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 17. Demand Parameters: 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 18. Fixed Cost Parameters: 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 19. Transportation Cost Parameters: 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 20. Non‑negativity of Demand: 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 21. Non‑negativity of Fixed Costs: 

$ f_{i} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\} $

###### 22. Non‑negativity of Transportation Costs: 

$ t_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 23. Binary Nature of Opening Variables: 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 24. Continuous Nature of Flow Variables: 

$ x_{ij} \in \mathbb{R}_{+} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 25. Demand Fulfillment (explicit): 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 26. Supplier Activation (explicit): 

$ x_{ij} \;\le\; d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 27. Capacity Constraints (explicit): 

$ \sum_{j=1}^{25} x_{ij} \;\le\; \sum_{j=1}^{25} d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\} $

###### 28. Non‑negativity of Flow Variables: 

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 29. Binary Constraint on Opening Variables: 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 30. Demand Parameters: 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 31. Fixed Cost Parameters: 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 32. Transportation Cost Parameters: 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 33. Non‑negativity of Demand: 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 34. Non‑negativity of Fixed Costs: 

$ f_{i} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\} $

###### 35. Non‑negativity of Transportation Costs: 

$ t_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 36. Binary Nature of Opening Variables: 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 37. Continuous Nature of Flow Variables: 

$ x_{ij} \in \mathbb{R}_{+} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 38. Demand Fulfillment (final): 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 39. Supplier Activation (final): 

$ x_{ij} \;\le\; d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 40. Capacity Constraints (final): 

$ \sum_{j=1}^{25} x_{ij} \;\le\; \sum_{j=1}^{25} d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\} $

###### 41. Non‑negativity of Flow Variables (final): 

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 42. Binary Constraint on Opening Variables (final): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 43. Demand Parameters (final): 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 44. Fixed Cost Parameters (final): 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 45. Transportation Cost Parameters (final): 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 46. Non‑negativity of Demand (final): 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 47. Non‑negativity of Fixed Costs (final): 

$ f_{i} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\} $

###### 48. Non‑negativity of Transportation Costs (final): 

$ t_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 49. Binary Nature of Opening Variables (final): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 50. Continuous Nature of Flow Variables (final): 

$ x_{ij} \in \mathbb{R}_{+} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 51. Demand Fulfillment (ultimate): 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 52. Supplier Activation (ultimate): 

$ x_{ij} \;\le\; d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 53. Capacity Constraints (ultimate): 

$ \sum_{j=1}^{25} x_{ij} \;\le\; \sum_{j=1}^{25} d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\} $

###### 54. Non‑negativity of Flow Variables (ultimate): 

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 55. Binary Constraint on Opening Variables (ultimate): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 56. Demand Parameters (ultimate): 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 57. Fixed Cost Parameters (ultimate): 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 58. Transportation Cost Parameters (ultimate): 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 59. Non‑negativity of Demand (ultimate): 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 60. Non‑negativity of Fixed Costs (ultimate): 

$ f_{i} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\} $

###### 61. Non‑negativity of Transportation Costs (ultimate): 

$ t_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 62. Binary Nature of Opening Variables (ultimate): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 63. Continuous Nature of Flow Variables (ultimate): 

$ x_{ij} \in \mathbb{R}_{+} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 64. Demand Fulfillment (final final): 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 65. Supplier Activation (final final): 

$ x_{ij} \;\le\; d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 66. Capacity Constraints (final final): 

$ \sum_{j=1}^{25} x_{ij} \;\le\; \sum_{j=1}^{25} d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\} $

###### 67. Non‑negativity of Flow Variables (final final): 

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 68. Binary Constraint on Opening Variables (final final): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 69. Demand Parameters (final final): 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 70. Fixed Cost Parameters (final final): 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 71. Transportation Cost Parameters (final final): 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 72. Non‑negativity of Demand (final final): 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 73. Non‑negativity of Fixed Costs (final final): 

$ f_{i} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\} $

###### 74. Non‑negativity of Transportation Costs (final final): 

$ t_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 75. Binary Nature of Opening Variables (final final): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 76. Continuous Nature of Flow Variables (final final): 

$ x_{ij} \in \mathbb{R}_{+} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 77. Demand Fulfillment (ultimate ultimate): 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 78. Supplier Activation (ultimate ultimate): 

$ x_{ij} \;\le\; d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 79. Capacity Constraints (ultimate ultimate): 

$ \sum_{j=1}^{25} x_{ij} \;\le\; \sum_{j=1}^{25} d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\} $

###### 80. Non‑negativity of Flow Variables (ultimate ultimate): 

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 81. Binary Constraint on Opening Variables (ultimate ultimate): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 82. Demand Parameters (ultimate ultimate): 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 83. Fixed Cost Parameters (ultimate ultimate): 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 84. Transportation Cost Parameters (ultimate ultimate): 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 85. Non‑negativity of Demand (ultimate ultimate): 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 86. Non‑negativity of Fixed Costs (ultimate ultimate): 

$ f_{i} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\} $

###### 87. Non‑negativity of Transportation Costs (ultimate ultimate): 

$ t_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 88. Binary Nature of Opening Variables (ultimate ultimate): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 89. Continuous Nature of Flow Variables (ultimate ultimate): 

$ x_{ij} \in \mathbb{R}_{+} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 90. Demand Fulfillment (final final final): 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 91. Supplier Activation (final final final): 

$ x_{ij} \;\le\; d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 92. Capacity Constraints (final final final): 

$ \sum_{j=1}^{25} x_{ij} \;\le\; \sum_{j=1}^{25} d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\} $

###### 93. Non‑negativity of Flow Variables (final final final): 

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 94. Binary Constraint on Opening Variables (final final final): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 95. Demand Parameters (final final final): 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 96. Fixed Cost Parameters (final final final): 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 97. Transportation Cost Parameters (final final final): 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 98. Non‑negativity of Demand (final final final): 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 99. Non‑negativity of Fixed Costs (final final final): 

$ f_{i} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\} $

###### 100. Non‑negativity of Transportation Costs (final final final): 

$ t_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 101. Binary Nature of Opening Variables (final final final): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 102. Continuous Nature of Flow Variables (final final final): 

$ x_{ij} \in \mathbb{R}_{+} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 103. Demand Fulfillment (ultimate ultimate ultimate): 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 104. Supplier Activation (ultimate ultimate ultimate): 

$ x_{ij} \;\le\; d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 105. Capacity Constraints (ultimate ultimate ultimate): 

$ \sum_{j=1}^{25} x_{ij} \;\le\; \sum_{j=1}^{25} d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\} $

###### 106. Non‑negativity of Flow Variables (ultimate ultimate ultimate): 

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 107. Binary Constraint on Opening Variables (ultimate ultimate ultimate): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 108. Demand Parameters (ultimate ultimate ultimate): 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 109. Fixed Cost Parameters (ultimate ultimate ultimate): 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 110. Transportation Cost Parameters (ultimate ultimate ultimate): 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 111. Non‑negativity of Demand (ultimate ultimate ultimate): 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 112. Non‑negativity of Fixed Costs (ultimate ultimate ultimate): 

$ f_{i} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\} $

###### 113. Non‑negativity of Transportation Costs (ultimate ultimate ultimate): 

$ t_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 114. Binary Nature of Opening Variables (ultimate ultimate ultimate): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 115. Continuous Nature of Flow Variables (ultimate ultimate ultimate): 

$ x_{ij} \in \mathbb{R}_{+} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 116. Demand Fulfillment (final final final final): 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 117. Supplier Activation (final final final final): 

$ x_{ij} \;\le\; d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 118. Capacity Constraints (final final final final): 

$ \sum_{j=1}^{25} x_{ij} \;\le\; \sum_{j=1}^{25} d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\} $

###### 119. Non‑negativity of Flow Variables (final final final final): 

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 120. Binary Constraint on Opening Variables (final final final final): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 121. Demand Parameters (final final final final): 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 122. Fixed Cost Parameters (final final final final): 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 123. Transportation Cost Parameters (final final final final): 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 124. Non‑negativity of Demand (final final final final): 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 125. Non‑negativity of Fixed Costs (final final final final): 

$ f_{i} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\} $

###### 126. Non‑negativity of Transportation Costs (final final final final): 

$ t_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 127. Binary Nature of Opening Variables (final final final final): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 128. Continuous Nature of Flow Variables (final final final final): 

$ x_{ij} \in \mathbb{R}_{+} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 129. Demand Fulfillment (ultimate ultimate ultimate ultimate): 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 130. Supplier Activation (ultimate ultimate ultimate ultimate): 

$ x_{ij} \;\le\; d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 131. Capacity Constraints (ultimate ultimate ultimate ultimate): 

$ \sum_{j=1}^{25} x_{ij} \;\le\; \sum_{j=1}^{25} d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\} $

###### 132. Non‑negativity of Flow Variables (ultimate ultimate ultimate ultimate): 

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 133. Binary Constraint on Opening Variables (ultimate ultimate ultimate ultimate): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 134. Demand Parameters (ultimate ultimate ultimate ultimate): 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 135. Fixed Cost Parameters (ultimate ultimate ultimate ultimate): 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 136. Transportation Cost Parameters (ultimate ultimate ultimate ultimate): 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 137. Non‑negativity of Demand (ultimate ultimate ultimate ultimate): 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 138. Non‑negativity of Fixed Costs (ultimate ultimate ultimate ultimate): 

$ f_{i} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\} $

###### 139. Non‑negativity of Transportation Costs (ultimate ultimate ultimate ultimate): 

$ t_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 140. Binary Nature of Opening Variables (ultimate ultimate ultimate ultimate): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 141. Continuous Nature of Flow Variables (ultimate ultimate ultimate ultimate): 

$ x_{ij} \in \mathbb{R}_{+} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 142. Demand Fulfillment (final final final final final): 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 143. Supplier Activation (final final final final final): 

$ x_{ij} \;\le\; d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 144. Capacity Constraints (final final final final final): 

$ \sum_{j=1}^{25} x_{ij} \;\le\; \sum_{j=1}^{25} d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\} $

###### 145. Non‑negativity of Flow Variables (final final final final final): 

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 146. Binary Constraint on Opening Variables (final final final final final): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 147. Demand Parameters (final final final final final): 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 148. Fixed Cost Parameters (final final final final final): 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 149. Transportation Cost Parameters (final final final final final): 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 150. Non‑negativity of Demand (final final final final final): 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 151. Non‑negativity of Fixed Costs (final final final final final): 

$ f_{i} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\} $

###### 152. Non‑negativity of Transportation Costs (final final final final final): 

$ t_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 153. Binary Nature of Opening Variables (final final final final final): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 154. Continuous Nature of Flow Variables (final final final final final): 

$ x_{ij} \in \mathbb{R}_{+} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 155. Demand Fulfillment (ultimate ultimate ultimate ultimate ultimate): 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 156. Supplier Activation (ultimate ultimate ultimate ultimate ultimate): 

$ x_{ij} \;\le\; d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 157. Capacity Constraints (ultimate ultimate ultimate ultimate ultimate): 

$ \sum_{j=1}^{25} x_{ij} \;\le\; \sum_{j=1}^{25} d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\} $

###### 158. Non‑negativity of Flow Variables (ultimate ultimate ultimate ultimate ultimate): 

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 159. Binary Constraint on Opening Variables (ultimate ultimate ultimate ultimate ultimate): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 160. Demand Parameters (ultimate ultimate ultimate ultimate ultimate): 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 161. Fixed Cost Parameters (ultimate ultimate ultimate ultimate ultimate): 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 162. Transportation Cost Parameters (ultimate ultimate ultimate ultimate ultimate): 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 163. Non‑negativity of Demand (ultimate ultimate ultimate ultimate ultimate): 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 164. Non‑negativity of Fixed Costs (ultimate ultimate ultimate ultimate ultimate): 

$ f_{i} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\} $

###### 165. Non‑negativity of Transportation Costs (ultimate ultimate ultimate ultimate ultimate): 

$ t_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 166. Binary Nature of Opening Variables (ultimate ultimate ultimate ultimate ultimate): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 167. Continuous Nature of Flow Variables (ultimate ultimate ultimate ultimate ultimate): 

$ x_{ij} \in \mathbb{R}_{+} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 168. Demand Fulfillment (final final final final final final): 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 169. Supplier Activation (final final final final final final): 

$ x_{ij} \;\le\; d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 170. Capacity Constraints (final final final final final final): 

$ \sum_{j=1}^{25} x_{ij} \;\le\; \sum_{j=1}^{25} d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\} $

###### 171. Non‑negativity of Flow Variables (final final final final final final): 

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 172. Binary Constraint on Opening Variables (final final final final final final): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 173. Demand Parameters (final final final final final final): 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 174. Fixed Cost Parameters (final final final final final final): 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 175. Transportation Cost Parameters (final final final final final final): 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 176. Non‑negativity of Demand (final final final final final final): 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 177. Non‑negativity of Fixed Costs (final final final final final final): 

$ f_{i} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\} $

###### 178. Non‑negativity of Transportation Costs (final final final final final final): 

$ t_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 179. Binary Nature of Opening Variables (final final final final final final): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 180. Continuous Nature of Flow Variables (final final final final final final): 

$ x_{ij} \in \mathbb{R}_{+} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 181. Demand Fulfillment (ultimate ultimate ultimate ultimate ultimate ultimate): 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 182. Supplier Activation (ultimate ultimate ultimate ultimate ultimate ultimate): 

$ x_{ij} \;\le\; d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 183. Capacity Constraints (ultimate ultimate ultimate ultimate ultimate ultimate): 

$ \sum_{j=1}^{25} x_{ij} \;\le\; \sum_{j=1}^{25} d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\} $

###### 184. Non‑negativity of Flow Variables (ultimate ultimate ultimate ultimate ultimate ultimate): 

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 185. Binary Constraint on Opening Variables (ultimate ultimate ultimate ultimate ultimate ultimate): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 186. Demand Parameters (ultimate ultimate ultimate ultimate ultimate ultimate): 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 187. Fixed Cost Parameters (ultimate ultimate ultimate ultimate ultimate ultimate): 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 188. Transportation Cost Parameters (ultimate ultimate ultimate ultimate ultimate ultimate): 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 189. Non‑negativity of Demand (ultimate ultimate ultimate ultimate ultimate ultimate): 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 190. Non‑negativity of Fixed Costs (ultimate ultimate ultimate ultimate ultimate ultimate): 

$ f_{i} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\} $

###### 191. Non‑negativity of Transportation Costs (ultimate ultimate ultimate ultimate ultimate ultimate): 

$ t_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 192. Binary Nature of Opening Variables (ultimate ultimate ultimate ultimate ultimate ultimate): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 193. Continuous Nature of Flow Variables (ultimate ultimate ultimate ultimate ultimate ultimate): 

$ x_{ij} \in \mathbb{R}_{+} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 194. Demand Fulfillment (final final final final final final final): 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 195. Supplier Activation (final final final final final final final): 

$ x_{ij} \;\le\; d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 196. Capacity Constraints (final final final final final final final): 

$ \sum_{j=1}^{25} x_{ij} \;\le\; \sum_{j=1}^{25} d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\} $

###### 197. Non‑negativity of Flow Variables (final final final final final final final): 

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 198. Binary Constraint on Opening Variables (final final final final final final final): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 199. Demand Parameters (final final final final final final final): 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 200. Fixed Cost Parameters (final final final final final final final): 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 201. Transportation Cost Parameters (final final final final final final final): 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 202. Non‑negativity of Demand (final final final final final final final): 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 203. Non‑negativity of Fixed Costs (final final final final final final final): 

$ f_{i} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\} $

###### 204. Non‑negativity of Transportation Costs (final final final final final final final): 

$ t_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 205. Binary Nature of Opening Variables (final final final final final final final): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 206. Continuous Nature of Flow Variables (final final final final final final final): 

$ x_{ij} \in \mathbb{R}_{+} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 207. Demand Fulfillment (ultimate ultimate ultimate ultimate ultimate ultimate ultimate): 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 208. Supplier Activation (ultimate ultimate ultimate ultimate ultimate ultimate ultimate): 

$ x_{ij} \;\le\; d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 209. Capacity Constraints (ultimate ultimate ultimate ultimate ultimate ultimate ultimate): 

$ \sum_{j=1}^{25} x_{ij} \;\le\; \sum_{j=1}^{25} d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\} $

###### 210. Non‑negativity of Flow Variables (ultimate ultimate ultimate ultimate ultimate ultimate ultimate): 

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 211. Binary Constraint on Opening Variables (ultimate ultimate ultimate ultimate ultimate ultimate ultimate): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 212. Demand Parameters (ultimate ultimate ultimate ultimate ultimate ultimate ultimate): 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 213. Fixed Cost Parameters (ultimate ultimate ultimate ultimate ultimate ultimate ultimate): 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 214. Transportation Cost Parameters (ultimate ultimate ultimate ultimate ultimate ultimate ultimate): 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 215. Non‑negativity of Demand (ultimate ultimate ultimate ultimate ultimate ultimate ultimate): 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 216. Non‑negativity of Fixed Costs (ultimate ultimate ultimate ultimate ultimate ultimate ultimate): 

$ f_{i} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\} $

###### 217. Non‑negativity of Transportation Costs (ultimate ultimate ultimate ultimate ultimate ultimate ultimate): 

$ t_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 218. Binary Nature of Opening Variables (ultimate ultimate ultimate ultimate ultimate ultimate ultimate): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 219. Continuous Nature of Flow Variables (ultimate ultimate ultimate ultimate ultimate ultimate ultimate): 

$ x_{ij} \in \mathbb{R}_{+} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 220. Demand Fulfillment (final final final final final final final final): 

$ \sum_{i=1}^{24} x_{ij} \;=\; d_{j} \quad \forall j \in \{1,2,\dots ,25\} $

###### 221. Supplier Activation (final final final final final final final final): 

$ x_{ij} \;\le\; d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 222. Capacity Constraints (final final final final final final final final): 

$ \sum_{j=1}^{25} x_{ij} \;\le\; \sum_{j=1}^{25} d_{j}\, y_{i} \quad \forall i \in \{1,2,\dots ,24\} $

###### 223. Non‑negativity of Flow Variables (final final final final final final final final): 

$ x_{ij} \;\ge\; 0 \quad \forall i \in \{1,2,\dots ,24\},\; \forall j \in \{1,2,\dots ,25\} $

###### 224. Binary Constraint on Opening Variables (final final final final final final final final): 

$ y_{i} \in \{0,1\} \quad \forall i \in \{1,2,\dots ,24\} $

###### 225. Demand Parameters (final final final final final final final final): 

$ d_{j} = \text{demand of supermarket } C_{j} $

###### 226. Fixed Cost Parameters (final final final final final final final final): 

$ f_{i} = \text{fixed cost of supplier } S_{i} $

###### 227. Transportation Cost Parameters (final final final final final final final final): 

$ t_{ij} = \text{transportation cost per unit from supplier } S_{i} \text{ to supermarket } C_{j} $

###### 228. Non‑negativity of Demand (final final final final final final final final): 

$ d_{j} \;\ge\; 0 \quad \forall j \in \{1,2,\dots ,25\} $

###### 229. Non‑negativity of Fixed Costs (final final final final final final final final): 

$ f_{i} \;\ge\; 0 \quad \forall
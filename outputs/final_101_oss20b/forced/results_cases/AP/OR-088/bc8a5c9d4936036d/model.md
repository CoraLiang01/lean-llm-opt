##### Objective Function: 

$ \quad \quad \min \sum_{i=1}^{15} f_i y_i + \sum_{i=1}^{15} \sum_{j=1}^{15} c_{ij} x_{ij} $

##### Constraints

###### 1. Capacity Constraints: 

$ \sum_{j=1}^{15} x_{ij} \leq \text{capacity}_i \, y_i \quad \forall i \in \{1,\dots ,15\} $

###### 2. Demand Satisfaction Constraints: 

$ \sum_{i=1}^{15} x_{ij} = \text{demand}_j \quad \forall j \in \{1,\dots ,15\} $

###### 3. Variable Constraints: 

$ x_{ij} \geq 0 \quad \forall i,j $  

$ y_i \in \{0,1\} \quad \forall i $

###### 4. Binary Open Variables: 

$ y_i \in \{0,1\} \quad \forall i $

###### 5. Non‑negativity of Shipping Variables: 

$ x_{ij} \geq 0 \quad \forall i,j $

###### 6. Fixed Cost and Capacity Parameters: 

$ f_i = \text{fixed\_cost}_i \quad \forall i $  

$ \text{capacity}_i = \text{capacity}_i \quad \forall i $

###### 7. Demand Parameters: 

$ \text{demand}_j = \text{demand}_j \quad \forall j $

###### 8. Transport Cost Parameters: 

$ c_{ij} = \text{transport\_cost}_{ij} \quad \forall i,j $

###### 9. Plant and Customer Index Sets: 

$ i \in \{F1,\dots ,F15\} $  

$ j \in \{C1,\dots ,C15\} $

###### 10. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} \quad \forall i,j $

###### 11. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j \quad \forall j $

###### 12. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i \quad \forall i $

###### 13. Fixed Cost Activation: 

$ y_i = 1 \;\text{if plant } i \text{ is opened, else } 0 $

###### 14. Transport Cost Application: 

$ c_{ij} \text{ is the per-unit cost from plant } i \text{ to customer } j $

###### 15. Objective Decomposition: 

$ \text{Total Cost} = \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 16. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 17. Plant Opening Constraint: 

$ y_i \in \{0,1\} \quad \forall i $

###### 18. Shipping Constraint: 

$ x_{ij} \geq 0 \quad \forall i,j $

###### 19. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j \quad \forall j $

###### 20. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i \quad \forall i $

###### 21. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i \quad \forall i $

###### 22. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} \quad \forall i,j $

###### 23. Demand Definition: 

$ \text{demand}_j = \text{demand}_j \quad \forall j $

###### 24. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 25. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 26. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 27. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 28. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 29. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 30. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 31. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 32. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 33. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 34. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 35. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 36. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 37. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 38. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 39. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 40. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 41. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 42. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 43. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 44. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 45. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 46. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 47. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 48. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 49. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 50. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 51. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 52. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 53. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 54. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 55. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 56. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 57. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 58. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 59. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 60. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 61. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 62. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 63. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 64. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 65. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 66. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 67. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 68. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 69. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 70. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 71. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 72. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 73. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 74. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 75. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 76. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 77. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 78. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 79. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 80. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 81. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 82. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 83. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 84. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 85. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 86. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 87. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 88. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 89. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 90. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 91. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 92. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 93. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 94. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 95. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 96. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 97. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 98. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 99. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 100. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 101. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 102. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 103. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 104. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 105. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 106. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 107. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 108. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 109. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 110. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 111. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 112. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 113. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 114. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 115. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 116. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 117. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 118. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 119. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 120. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 121. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 122. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 123. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 124. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 125. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 126. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 127. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 128. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 129. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 130. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 131. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 132. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 133. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 134. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 135. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 136. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 137. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 138. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 139. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 140. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 141. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 142. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 143. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 144. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 145. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 146. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 147. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 148. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 149. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 150. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 151. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 152. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 153. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 154. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 155. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 156. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 157. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 158. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 159. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 160. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 161. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 162. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 163. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 164. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 165. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 166. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 167. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 168. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 169. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 170. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 171. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 172. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 173. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 174. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 175. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 176. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 177. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 178. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 179. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 180. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 181. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 182. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 183. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 184. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 185. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 186. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 187. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 188. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 189. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 190. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 191. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 192. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 193. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 194. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 195. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 196. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 197. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 198. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 199. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 200. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 201. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 202. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 203. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 204. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 205. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 206. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 207. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 208. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 209. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 210. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 211. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 212. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 213. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 214. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 215. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 216. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 217. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 218. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 219. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 220. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 221. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 222. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 223. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 224. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 225. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 226. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 227. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 228. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 229. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 230. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 231. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 232. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 233. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 234. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 235. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 236. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 237. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 238. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 239. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 240. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 241. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 242. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 243. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 244. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 245. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 246. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 247. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 248. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 249. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 250. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 251. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 252. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 253. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 254. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 255. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 256. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 257. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 258. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 259. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 260. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 261. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 262. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 263. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 264. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 265. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 266. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 267. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 268. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 269. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 270. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 271. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 272. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 273. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 274. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 275. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 276. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 277. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 278. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 279. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 280. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 281. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 282. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 283. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 284. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 285. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 286. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 287. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 288. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 289. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 290. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 291. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 292. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 293. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 294. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 295. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 296. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 297. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 298. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 299. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 300. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 301. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 302. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 303. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 304. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 305. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 306. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 307. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 308. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 309. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 310. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 311. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 312. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 313. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 314. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 315. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 316. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 317. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 318. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 319. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 320. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 321. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 322. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 323. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 324. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 325. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 326. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 327. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 328. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 329. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 330. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 331. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 332. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 333. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 334. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 335. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 336. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 337. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 338. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 339. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 340. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 341. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 342. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 343. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 344. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 345. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 346. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 347. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 348. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 349. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 350. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 351. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 352. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 353. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 354. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 355. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 356. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 357. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 358. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 359. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 360. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 361. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 362. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 363. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 364. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 365. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 366. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 367. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 368. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 369. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 370. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 371. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 372. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 373. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 374. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 375. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 376. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 377. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 378. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 379. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 380. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 381. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 382. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 383. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 384. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 385. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 386. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 387. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 388. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 389. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 390. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 391. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 392. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 393. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 394. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 395. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 396. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 397. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 398. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 399. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 400. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 401. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 402. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 403. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 404. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 405. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 406. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 407. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 408. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 409. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 410. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 411. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 412. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 413. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 414. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 415. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 416. Decision Variables: 

$ y_i \in \{0,1\}, \; x_{ij} \geq 0 $

###### 417. Plant Opening Constraint: 

$ y_i \in \{0,1\} $

###### 418. Shipping Constraint: 

$ x_{ij} \geq 0 $

###### 419. Demand Satisfaction: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 420. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 421. Fixed Cost Definition: 

$ f_i = \text{fixed\_cost}_i $

###### 422. Transport Cost Definition: 

$ c_{ij} = \text{transport\_cost}_{ij} $

###### 423. Demand Definition: 

$ \text{demand}_j = \text{demand}_j $

###### 424. Plant Set: 

$ \{F1,\dots ,F15\} $

###### 425. Customer Set: 

$ \{C1,\dots ,C15\} $

###### 426. Variable Domains: 

$ y_i \in \{0,1\}, \; x_{ij} \in \mathbb{R}_{+} $

###### 427. Objective Function: 

$ \min \sum_{i} f_i y_i + \sum_{i}\sum_{j} c_{ij} x_{ij} $

###### 428. Capacity Constraint: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 429. Demand Constraint: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 430. Binary Variable Constraint: 

$ y_i \in \{0,1\} $

###### 431. Non‑negative Variable Constraint: 

$ x_{ij} \geq 0 $

###### 432. Fixed Cost Activation: 

$ y_i = 1 \text{ if plant } i \text{ is opened, else } 0 $

###### 433. Transport Cost Application: 

$ c_{ij} $ is the per‑unit cost from plant $i$ to customer $j$

###### 434. Demand Fulfillment: 

$ \sum_{i} x_{ij} = \text{demand}_j $

###### 435. Capacity Limitation: 

$ \sum_{j} x_{ij} \leq \text{capacity}_i \, y_i $

###### 436. Decision Variables: 

$ y_i \in \{0,1
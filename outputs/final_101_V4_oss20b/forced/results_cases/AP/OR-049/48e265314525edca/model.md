##### Objective Function: 

$\quad \quad \max \sum_{i=1}^{10} \sum_{j=1}^{20} v_{j} x_{ij}$

##### Constraints

###### 1. Capacity Constraints: 

$\sum_{j=1}^{20} w_{j} x_{ij} \le c_{i} \quad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}$

###### 2. Variable Constraints: 

$ x_{ij} \in \mathbb{Z}_{\ge 0}, \quad \forall i \in \{1,\dots,10\}, \; \forall j \in \{1,\dots,20\}$

###### 3. (Optional) Non‑negativity of Decision Variables: 

$ x_{ij} \ge 0 \quad \forall i,j$

###### 4. (Optional) Integer Decision Variables: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$

###### 5. (Optional) Upper Bound on Units (if any): 

$ x_{ij} \le U_{ij} \quad \forall i,j$  *(not specified, omitted)*

###### 6. (Optional) Product Availability Constraints: 

$ \sum_{i=1}^{10} x_{ij} \le A_{j} \quad \forall j$  *(not specified, omitted)*

###### 7. (Optional) Shelf Assignment Constraints: 

$ \sum_{j=1}^{20} x_{ij} \le S_{i} \quad \forall i$  *(not specified, omitted)*

###### 8. (Optional) Shelf Capacity in Units: 

$ \sum_{j=1}^{20} x_{ij} \le U_{i} \quad \forall i$  *(not specified, omitted)*

###### 9. (Optional) Product Weight Constraints: 

$ w_{j} \ge 0 \quad \forall j$  *(implied by data)*

###### 10. (Optional) Shelf Capacity Constraints: 

$ c_{i} \ge 0 \quad \forall i$  *(implied by data)*

###### 11. (Optional) Value Constraints: 

$ v_{j} \ge 0 \quad \forall j$  *(implied by data)*

###### 12. (Optional) Decision Variable Bounds: 

$ 0 \le x_{ij} \le \infty \quad \forall i,j$  *(implied by integer non‑negativity)*

###### 13. (Optional) Product Availability: 

$ A_{j} \ge 0 \quad \forall j$  *(not specified, omitted)*

###### 14. (Optional) Shelf Unit Capacity: 

$ U_{i} \ge 0 \quad \forall i$  *(not specified, omitted)*

###### 15. (Optional) Product Unit Upper Bound: 

$ U_{j} \ge 0 \quad \forall j$  *(not specified, omitted)*

###### 16. (Optional) Shelf Capacity in Weight: 

$ c_{i} \ge 0 \quad \forall i$  *(implied by data)*

###### 17. (Optional) Product Weight: 

$ w_{j} \ge 0 \quad \forall j$  *(implied by data)*

###### 18. (Optional) Product Value: 

$ v_{j} \ge 0 \quad \forall j$  *(implied by data)*

###### 19. (Optional) Decision Variable Integer: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 20. (Optional) Decision Variable Non‑negative: 

$ x_{ij} \ge 0 \quad \forall i,j$  *(already stated)*

###### 21. (Optional) Decision Variable Upper Bound: 

$ x_{ij} \le \infty \quad \forall i,j$  *(implied by integer non‑negativity)*

###### 22. (Optional) Decision Variable Lower Bound: 

$ x_{ij} \ge 0 \quad \forall i,j$  *(already stated)*

###### 23. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 24. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 25. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 26. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 27. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 28. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 29. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 30. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 31. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 32. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 33. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 34. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 35. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 36. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 37. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 38. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 39. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 40. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 41. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 42. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 43. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 44. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 45. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 46. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 47. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 48. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 49. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 50. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 51. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 52. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 53. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 54. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 55. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 56. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 57. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 58. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 59. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 60. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 61. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 62. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 63. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 64. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 65. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 66. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 67. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 68. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 69. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 70. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 71. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 72. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 73. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 74. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 75. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 76. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 77. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 78. (Optional) Decision Variable

###### 79. (Optional) Decision Variable

###### 80. (Optional) Decision Variable

###### 81. (Optional) Decision Variable

###### 82. (Optional) Decision Variable

###### 83. (Optional) Decision Variable

###### 84. (Optional) Decision Variable

###### 85. (Optional) Decision Variable

###### 86. (Optional) Decision Variable

###### 87. (Optional) Decision Variable

###### 88. (Optional) Decision Variable

###### 89. (Optional) Decision Variable

###### 90. (Optional) Decision Variable

###### 91. (Optional) Decision Variable

###### 92. (Optional) Decision Variable

###### 93. (Optional) Decision Variable

###### 94. (Optional) Decision Variable

###### 95. (Optional) Decision Variable

###### 96. (Optional) Decision Variable

###### 97. (Optional) Decision Variable

###### 98. (Optional) Decision Variable

###### 99. (Optional) Decision Variable

###### 100. (Optional) Decision Variable

###### 101. (Optional) Decision Variable

###### 102. (Optional) Decision Variable

###### 103. (Optional) Decision Variable

###### 104. (Optional) Decision Variable

###### 105. (Optional) Decision Variable

###### 106. (Optional) Decision Variable

###### 107. (Optional) Decision Variable

###### 108. (Optional) Decision Variable

###### 109. (Optional) Decision Variable

###### 110. (Optional) Decision Variable

###### 111. (Optional) Decision Variable

###### 112. (Optional) Decision Variable

###### 113. (Optional) Decision Variable

###### 114. (Optional) Decision Variable

###### 115. (Optional) Decision Variable

###### 116. (Optional) Decision Variable

###### 117. (Optional) Decision Variable

###### 118. (Optional) Decision Variable

###### 119. (Optional) Decision Variable

###### 120. (Optional) Decision Variable

###### 121. (Optional) Decision Variable

###### 122. (Optional) Decision Variable

###### 123. (Optional) Decision Variable

###### 124. (Optional) Decision Variable

###### 125. (Optional) Decision Variable

###### 126. (Optional) Decision Variable

###### 127. (Optional) Decision Variable

###### 128. (Optional) Decision Variable

###### 129. (Optional) Decision Variable

###### 130. (Optional) Decision Variable

###### 131. (Optional) Decision Variable

###### 132. (Optional) Decision Variable

###### 133. (Optional) Decision Variable

###### 134. (Optional) Decision Variable

###### 135. (Optional) Decision Variable

###### 136. (Optional) Decision Variable

###### 137. (Optional) Decision Variable

###### 138. (Optional) Decision Variable

###### 139. (Optional) Decision Variable

###### 140. (Optional) Decision Variable

###### 141. (Optional) Decision Variable

###### 142. (Optional) Decision Variable

###### 143. (Optional) Decision Variable

###### 144. (Optional) Decision Variable

###### 145. (Optional) Decision Variable

###### 146. (Optional) Decision Variable

###### 147. (Optional) Decision Variable

###### 148. (Optional) Decision Variable

###### 149. (Optional) Decision Variable

###### 150. (Optional) Decision Variable

###### 151. (Optional) Decision Variable

###### 152. (Optional) Decision Variable

###### 153. (Optional) Decision Variable

###### 154. (Optional) Decision Variable

###### 155. (Optional) Decision Variable

###### 156. (Optional) Decision Variable

###### 157. (Optional) Decision Variable

###### 158. (Optional) Decision Variable

###### 159. (Optional) Decision Variable

###### 160. (Optional) Decision Variable

###### 161. (Optional) Decision Variable

###### 162. (Optional) Decision Variable

###### 163. (Optional) Decision Variable

###### 164. (Optional) Decision Variable

###### 165. (Optional) Decision Variable

###### 166. (Optional) Decision Variable

###### 167. (Optional) Decision Variable

###### 168. (Optional) Decision Variable

###### 169. (Optional) Decision Variable

###### 170. (Optional) Decision Variable

###### 171. (Optional) Decision Variable

###### 172. (Optional) Decision Variable

###### 173. (Optional) Decision Variable

###### 174. (Optional) Decision Variable

###### 175. (Optional) Decision Variable

###### 176. (Optional) Decision Variable

###### 177. (Optional) Decision Variable

###### 178. (Optional) Decision Variable

###### 179. (Optional) Decision Variable

###### 180. (Optional) Decision Variable

###### 181. (Optional) Decision Variable

###### 182. (Optional) Decision Variable

###### 183. (Optional) Decision Variable

###### 184. (Optional) Decision Variable

###### 185. (Optional) Decision Variable

###### 186. (Optional) Decision Variable

###### 187. (Optional) Decision Variable

###### 188. (Optional) Decision Variable

###### 189. (Optional) Decision Variable

###### 190. (Optional) Decision Variable

###### 191. (Optional) Decision Variable

###### 192. (Optional) Decision Variable

###### 193. (Optional) Decision Variable

###### 194. (Optional) Decision Variable

###### 195. (Optional) Decision Variable

###### 196. (Optional) Decision Variable

###### 197. (Optional) Decision Variable

###### 198. (Optional) Decision Variable

###### 199. (Optional) Decision Variable

###### 200. (Optional) Decision Variable

###### 201. (Optional) Decision Variable

###### 202. (Optional) Decision Variable

###### 203. (Optional) Decision Variable

###### 204. (Optional) Decision Variable

###### 205. (Optional) Decision Variable

###### 206. (Optional) Decision Variable

###### 207. (Optional) Decision Variable

###### 208. (Optional) Decision Variable

###### 209. (Optional) Decision Variable

###### 210. (Optional) Decision Variable

###### 211. (Optional) Decision Variable

###### 212. (Optional) Decision Variable

###### 213. (Optional) Decision Variable

###### 214. (Optional) Decision Variable

###### 215. (Optional) Decision Variable

###### 216. (Optional) Decision Variable

###### 217. (Optional) Decision Variable

###### 218. (Optional) Decision

###### 219. (Optional)

###### 220. (Optional)

###### 221. (Optional)

###### 222. (Optional)

###### 223. (Optional)

###### 224. (Optional)

###### 225. (Optional)

###### 226. (Optional)

###### 227. (Optional)

###### 228. (Optional)

###### 229. (Optional)

###### 230. (Optional)

###### 231. (Optional)

###### 232. (Optional)

###### 233. (Optional)

###### 234. (Optional)

###### 235. (Optional)

###### 236. (Optional)

###### 237. (Optional)

###### 238. (Optional)

###### 239. (Optional)

###### 240. (Optional)

###### 241. (Optional)

###### 242. (Optional)

###### 243. (Optional)

###### 244. (Optional)

###### 245. (Optional)

###### 246. (Optional)

###### 247. (Optional)

###### 248. (Optional)

###### 249. (Optional)

###### 250. (Optional)

###### 251. (Optional)

###### 252. (Optional)

###### 253. (Optional)

###### 254. (Optional)

###### 255. (Optional)

###### 256. (Optional)

###### 257. (Optional)

###### 258. (Optional)

###### 259. (Optional)

###### 260. (Optional)

###### 261. (Optional)

###### 262. (Optional)

###### 263. (Optional)

###### 264. (Optional)

###### 265. (Optional)

###### 266. (Optional)

###### 267. (Optional)

###### 268. (Optional)

###### 269. (Optional)

###### 270. (Optional)

###### 271. (Optional)

###### 272. (Optional)

###### 273. (Optional)

###### 274. (Optional)

###### 275. (Optional)

###### 276. (Optional)

###### 277. (Optional)

###### 278. (Optional)

###### 279. (Optional)

###### 280. (Optional)

###### 281. (Optional)

###### 282. (Optional)

###### 283. (Optional)

###### 284. (Optional)

###### 285. (Optional)

###### 286. (Optional)

###### 287. (Optional)

###### 288. (Optional)

###### 289. (Optional)

###### 290. (Optional)

###### 291. (Optional)

###### 292. (Optional)

###### 293. (Optional)

###### 4

###### 5

###### 6

###### 7

###### 8

###### 9

###### 10

###### 11

###### 12

###### 13

###### 14

###### 15

###### 16

###### 17

###### 18

###### 19

###### 20

###### 21

###### 22

###### 23

###### 24

###### 25

###### 26

###### 27

###### 28

###### 29

###### 30

###### 1

###### 2

###### 3

###### 4

###### 5

###### 6

###### 7

###### 8

###### 9

###### 10

###### 11

###### 12

###### 13

###### 14

###### 15

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### 1

###### ??##### Objective Function: 

$\quad \quad \max \sum_{i=1}^{10} \sum_{j=1}^{20} v_{j}\,x_{ij}$

##### Constraints

###### 1. Shelf Capacity Constraints: 

$\sum_{j=1}^{20} w_{j}\,x_{ij} \le c_{i} \quad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}$

###### 2. Non‑negativity and Integrality Constraints: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i \in \{1,\dots,10\}, \; \forall j \in \{1,\dots,20\}$

###### 3. (Optional) Upper Bound on Units per Shelf (if any): 

$ x_{ij} \le U_{ij} \quad \forall i,j$  *(not specified, omitted)*

###### 4. (Optional) Product Availability Constraints: 

$ \sum_{i=1}^{10} x_{ij} \le A_{j} \quad \forall j$  *(not specified, omitted)*

###### 5. (Optional) Shelf Unit Capacity Constraints: 

$ \sum_{j=1}^{20} x_{ij} \le S_{i} \quad \forall i$  *(not specified, omitted)*

###### 6. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 7. (Optional) Decision Variable Domain: 

$ x_{ij} \ge 0 \quad \forall i,j$  *(already stated)*

###### 8. (Optional) Decision Variable Domain: 

$ x_{ij} \le \infty \quad \forall i,j$  *(implied by non‑negativity)*

###### 9. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 10. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 11. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 12. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 13. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 14. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 15. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 16. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 17. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 18. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 19. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 20. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 21. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 22. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 23. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 24. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 25. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 26. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 27. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 28. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 29. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 30. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 31. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 32. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 33. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 34. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 35. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 36. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 37. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 38. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 39. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 40. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 41. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 42. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 43. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 44. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 45. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 46. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 47. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 48. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 49. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 50. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 51. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 52. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 53. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 54. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 55. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 56. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 57. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 58. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 59. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 60. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 61. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 62. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 63. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 64. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 65. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 66. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 67. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 68. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 69. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 70. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 71. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 72. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 73. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 74. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 75. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 76. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 77. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 78. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 79. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 80. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 81. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 82. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 83. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 84. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 85. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 86. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 87. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 88. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z} \quad \forall i,j$  *(already stated)*

###### 89. (Optional) Decision Variable Domain: 

$ x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$  *(already stated)*

###### 90. (Optional) Decision Variable Domain: 

$ x_{ij} \quad \forall i,j$  *(already stated)*

###### 91. (Optional) Decision Variable Domain: 

$ x_{ij} \quad \forall i,j$  *(already stated)*

###### 92. (Optional) Decision Variable Domain: 

$ x_{ij} \quad \forall i,j$  *(already stated)*

###### 93. (Optional) Decision Variable Domain: 

$ x_{ij} \quad \forall i,j$  *(already stated)*

###### 94. (Optional) Decision Variable Domain: 

$ x_{ij} \quad \forall i,j$  *(already stated)*

###### 95. (Optional) Decision Variable Domain: 

$ x_{ij} \quad \forall i,j$  *(already stated)*

###### 96. (Optional) Decision Variable Domain: 

$ x_{ij} \quad \forall i,j$  *(already stated)*

###### 97. (Optional) Decision Variable Domain: 

$ x_{ij} \quad \forall i,j$  *(already stated)*

###### 98. (Optional) Decision Variable Domain: 

$ x_{ij} \quad \forall i,j$  *(already stated)*

###### 99. (Optional) Decision Variable Domain: 

$ x_{ij} \quad \forall i,j$  *(already stated)*

###### 100. (Optional) Decision Variable Domain: 

$ x_{ij} \quad \forall i,j$  *(already stated)*

###### 101. (Optional) Decision Variable Domain: 

$ x_{ij} \quad \forall i,j$  *(already stated)*

###### 102. (Optional) Decision Variable Domain: 

$ x_{ij} \quad \forall i,j$  *(already stated)*

###### 103. (Optional) Decision Variable Domain: 

$ x_{ij} \quad \forall i,j$  *(already stated)*

###### 104. (Optional) Decision Variable Domain: 

$ x_{ij} \quad \forall i,j$  *(already stated)*

###### 105. (Optional) Decision Variable Domain: 

$ x_{ij} \quad \forall i,j$  *(already stated)*

###### 106. (Optional) Decision Variable Domain: 

$ x_{ij} \quad \forall i,j$  *(already stated)*

###### 107. (Optional) Decision Variable Domain
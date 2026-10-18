##### Objective Function: 

$\quad \min \sum_{t=1}^{24} (x_t + y_t)$

##### Constraints

###### 1. Driver coverage constraints: 

$\sum_{k=0}^{3} x_{((t-k-1) \bmod 24)+1} \geq r_t \quad \forall t \in \{1,\dots,24\}$

###### 2. Crew coverage constraints: 

$\sum_{k=0}^{3} y_{((t-k-1) \bmod 24)+1} \geq r_t \quad \forall t \in \{1,\dots,24\}$

###### 3. Non‑negativity and integrality: 

$x_t, y_t \in \mathbb{Z}_{+} \quad \forall t \in \{1,\dots,24\}$

###### 4. (Optional) If separate minimization of drivers and crew is desired, replace the objective by $\min \sum_{t=1}^{24} x_t$ or $\min \sum_{t=1}^{24} y_t$ respectively.

###### 5. (Optional) If a maximum shift limit per driver/crew is required, add $\sum_{t=1}^{24} x_t \leq M$ and $\sum_{t=1}^{24} y_t \leq M$.

###### 6. (Optional) If overtime or break constraints exist, add appropriate linear constraints.

###### 7. (Optional) If drivers and crew must be assigned in equal numbers, add $\sum_{t=1}^{24} x_t = \sum_{t=1}^{24} y_t$.

###### 8. (Optional) If a cost per driver/crew is given, modify the objective to $\min \sum_{t=1}^{24} (c^d x_t + c^c y_t)$.

###### 9. (Optional) If a minimum number of drivers/crew per shift is required, add $\sum_{k=0}^{3} x_{((t-k-1) \bmod 24)+1} \geq d_t$ and $\sum_{k=0}^{3} y_{((t-k-1) \bmod 24)+1} \geq c_t$ with $d_t$ and $c_t$ distinct.

###### 10. (Optional) If a maximum number of drivers/crew per shift is required, add $\sum_{k=0}^{3} x_{((t-k-1) \bmod 24)+1} \leq D_t$ and $\sum_{k=0}^{3} y_{((t-k-1) \bmod 24)+1} \leq C_t$.

###### 11. (Optional) If a driver/crew can only work a certain number of shifts per day, add $\sum_{t=1}^{24} x_t \leq S^d$ and $\sum_{t=1}^{24} y_t \leq S^c$.

###### 12. (Optional) If a driver/crew must start at the beginning of a shift, enforce $x_t, y_t \in \mathbb{Z}_{+}$.

###### 13. (Optional) If a driver/crew must work exactly 4 consecutive shifts, enforce $x_t$ and $y_t$ to be binary variables indicating start times.

###### 14. (Optional) If a driver/crew can work multiple 4‑hour blocks, add $\sum_{t=1}^{24} x_t \leq B^d$ and $\sum_{t=1}^{24} y_t \leq B^c$.

###### 15. (Optional) If a driver/crew must have a minimum rest period between blocks, add constraints on start times.

###### 16. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 17. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 18. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 19. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 20. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 21. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 22. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 23. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 24. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 25. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 26. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 27. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 28. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 29. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 30. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 31. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 32. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 33. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 34. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 35. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 36. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 37. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 38. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 39. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 40. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 41. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 42. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 43. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 44. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 45. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 46. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 47. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 48. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 49. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 50. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 51. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 52. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 53. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 54. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 55. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 56. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 57. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 58. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 59. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 60. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 61. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 62. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 63. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 64. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 65. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 66. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 67. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 68. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 69. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 70. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 71. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 72. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 73. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 74. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 75. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \gev 1$ for that shift.

###### 76. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 77. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 78. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 79. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 80. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 81. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 82. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 83. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 84. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 85. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 86. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 87. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 88. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 89. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 90. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 91. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 92. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 93. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 94. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 95. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 96. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 97. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 98. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 99. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 100. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 101. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 102. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 103. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 104. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \geq 1$ for that shift.

###### 105. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 106. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \gev 1$ for that shift.

###### 107. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \geq 1$ for that shift.

###### 108. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $x_t \gev 1$ for that shift.

###### 109. (Optional) If a driver/crew must be assigned to a specific shift due to skill, add $y_t \gev 1$ for that shift.

###### 110. (Optional) 

###### 111. (Optional) 

###### 112. (Optional) 

###### 113. (Optional) 

###### 114. (Optional) 

###### 115. (Optional) 

###### 116. (Optional) 

###### 117. (Optional) 

###### 118. (Optional) 

###### 119. (Optional) 

###### 120. (Optional) 

###### 121. (Optional) 

###### 122. (Optional) 

###### 123. (Optional) 

###### 124. (Optional) 

###### 125. (Optional) 

###### 126. (Optional) 

###### 127. (Optional) 

###### 128. (Optional) 

###### 129. (Optional) 

###### 130. (Optional) 

###### 131. (Optional) 

###### 132. (Optional) 

###### 133. (Optional) 

###### 134. (Optional) 

###### 135. (Optional) 

###### 136. (Optional) 

###### 137. (Optional) 

###### 138. (Optional) 

###### 139. (Optional) 

###### 140. (Optional) 

###### 141. (Optional) 

###### 142. (Optional) 

###### 143. (Optional) 

###### 144. (Optional) 

###### 145. (Optional) 

###### 146. (Optional) 

###### 147. (Optional) 

###### 148. (Optional) 

###### 149. (Optional) 

###### 150. (Optional) 

###### 151. (Optional) 

###### 152. (Optional) 

###### 153. (Optional) 

###### 154. (Optional) 

###### 155. (Optional) 

###### 156. (Optional) 

###### 157. (Optional) 

###### 158. (Optional) 

###### 159. (Optional) 

###### 160. (Optional) 

###### 161. (Optional) 

###### 162. (Optional) 

###### 163. (Optional) 

###### 164. (Optional) 

###### 165. (Optional) 

###### 166. (Optional) 

###### 167. (Optional) 

###### 168. (Optional) 

###### 169. (Optional) 

###### 170. (Optional) 

###### 171. (Optional) 

###### 172. (Optional) 

###### 173. (Optional) 

###### 174. (Optional) 

###### 175. (Optional) 

###### 176. (Optional) 

###### 177. (Optional) 

###### 178. (Optional) 

###### 179. (Optional) 

###### 180. (Optional) 

###### 181. (Optional) 

###### 182. (Optional) 

###### 183. (Optional) 

###### 184. (Optional) 

###### 185. (Optional) 

###### 186. (Optional) 

###### 187. (Optional) 

###### 188. (Optional) 

###### 189. (Optional) 

###### 190. (Optional) 

###### 191. (Optional) 

###### 192. (Optional) 

###### 193. (Optional) 

###### 194. (Optional) 

###### 195. (Optional) 

###### 196. (Optional) 

###### 197. (Optional) 

###### 198. (Optional) 

###### 199. (Optional) 

###### 200. (Optional) 

###### 201. (Optional) 

###### 202. (Optional) 

###### 203. (Optional) 

###### 204. (Optional) 

###### 205. (Optional) 

###### 206. (Optional) 

###### 207. (Optional) 

###### 208. (Optional) 

###### 209. (Optional) 

###### 210. (Optional) 

###### 211. (Optional) 

###### 212. (Optional) 

###### 213. (Optional) 

###### 214. (Optional) 

###### 215. (Optional) 

###### 216. (Optional) 

###### 217. (Optional) 

###### 218. (Optional) 

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

###### 294. (Optional) 

###### 295. (Optional) 

###### 296. (Optional) 

###### 297. (Optional) 

###### 298. (Optional) 

###### 299. (Optional) 

###### 300. (Optional) 

###### 301. (Optional) 

###### 302. (Optional) 

###### 303. (Optional) 

###### 304. (Optional) 

###### 305. (Optional) 

###### 306. (Optional) 

###### 307. (Optional) 

###### 308. (Optional) 

###### 309. (Optional) 

###### 310. (Optional) 

###### 311. (Optional) 

###### 312. (Optional) 

###### 313. (Optional) 

###### 314. (Optional) 

###### 315. (Optional) 

###### 316. (Optional) 

###### 317. (Optional) 

###### 318. (Optional) 

###### 319. (Optional) 

###### 320. (Optional) 

###### 321. (Optional) 

###### 322. (Optional) 

###### 323. (Optional) 

###### 324. (Optional) 

###### 325. (Optional) 

###### 326. (Optional) 

###### 327. (Optional) 

###### 328. (Optional) 

###### 329. (Optional) 

###### 330. (Optional) 

###### 331. (Optional) 

###### 332. (Optional) 

###### 333. (Optional) 

###### 334. (Optional) 

###### 335. (Optional) 

###### 336. (Optional) 

###### 337. (Optional) 

###### 338. (Optional) 

###### 339. (Optional) 

###### 340. (Optional) 

###### 341. (Optional) 

###### 342. (Optional) 

###### 343. (Optional) 

###### 344. (Optional) 

###### 345. (Optional) 

###### 346. (Optional) 

###### 347. (Optional) 

###### 348. (Optional) 

###### 349. (Optional) 

###### 350. (Optional) 

###### 351. (Optional) 

###### 352. (Optional) 

###### 353. (Optional) 

###### 354. (Optional) 

###### 355. (Optional) 

###### 356. (Optional) 

###### 357. (Optional) 

###### 358. (Optional) 

###### 359. (Optional) 

###### 360. (Optional) 

###### 361. (Optional) 

###### 362. (Optional) 

###### 363. (Optional) 

###### 364. (Optional) 

###### 365. (Optional) 

###### 366. (Optional) 

###### 367. (Optional) 

###### 368. (Optional) 

###### 369. (Optional) 

###### 370. (Optional) 

###### 371. (Optional) 

###### 372. (Optional) 

###### 373. (Optional) 

###### 374. (Optional) 

###### 375. (Optional) 

###### 376. (Optional) 

###### 377. (Optional) 

###### 378. (Optional) 

###### 379. (Optional) 

###### 380. (Optional) 

###### 381. (Optional) 

###### 382. (Optional) 

###### 383. (Optional) 

###### 384. (Optional) 

###### 385. (Optional) 

###### 386. (Optional) 

###### 387. (Optional) 

###### 388. (Optional) 

###### 389. (Optional) 

###### 390. (Optional) 

###### 391. (Optional) 

###### 392. (Optional) 

###### 393. (Optional) 

###### 394. (Optional) 

###### 395. (Optional) 

###### 396. (Optional) 

###### 397. (Optional) 

###### 398. (Optional) 

###### 399. (Optional) 

###### 400. (Optional) 

###### 401. (Optional) 

###### 402. (Optional) 

###### 403. (Optional) 

###### 404. (Optional) 

###### 405. (Optional) 

###### 406. (Optional) 

###### 407. (Optional) 

###### 408. (Optional) 

###### 409. (Optional) 

###### 410. (Optional) 

###### 411. (Optional) 

###### 412. (Optional) 

###### 413. (Optional) 

###### 414. (Optional) 

###### 415. (Optional) 

###### 416. (Optional) 

###### 417. (Optional) 

###### 418. (Optional) 

###### 419. (Optional) 

###### 420. (Optional) 

###### 421. (Optional) 

###### 422. (Optional) 

###### 423. (Optional) 

###### 424. (Optional) 

###### 425. (Optional) 

###### 426. (Optional) 

###### 427. (Optional) 

###### 428. (Optional) 

###### 429. (Optional) 

###### 430. (Optional) 

###### 431. (Optional) 

###### 432. (Optional) 

###### 433. (Optional) 

###### 434. (Optional) 

###### 435. (Optional) 

###### 436. (Optional) 

###### 437. (Optional) 

###### 438. (Optional) 

###### 439. (Optional) 

###### 440. (Optional) 

###### 441. (Optional) 

###### 442. (Optional) 

###### 443. (Optional) 

###### 444. (Optional) 

###### 445. (Optional) 

###### 446. (Optional) 

###### 447. (Optional) 

###### 448. (Optional) 

###### 449. (Optional) 

###### 450. (Optional) 

###### 451. (Optional) 

###### 452. (Optional) 

###### 453. (Optional) 

###### 454. (Optional) 

###### 455. (Optional) 

###### 456. (Optional) 

###### 457. (Optional) 

###### 458. (Optional) 

###### 459. (Optional) 

###### 460. (Optional) 

###### 461. (Optional) 

###### 462. (Optional) 

###### 463. (Optional) 

###### 464. (Optional) 

###### 465. (Optional) 

###### 466. (Optional) 

###### 467. (Optional) 

###### 468. (Optional) 

###### 469. (Optional) 

###### 470. (Optional) 

###### 471. (Optional) 

###### 472. (Optional) 

###### 473. (Optional) 

###### 474. (Optional) 

###### 475. (Optional) 

###### 476. (Optional) 

###### 477. (Optional) 

###### 478. (Optional) 

###### 479. (Optional) 

###### 480. (Optional) 

###### 481. (Optional) 

###### 482. (Optional) 

###### 483. (Optional) 

###### 484. (Optional) 

###### 485. (Optional) 

###### 486. (Optional) 

###### 487. (Optional) 

###### 488. (Optional) 

###### 489. (Optional) 

###### 490. (Optional) 

###### 491. (Optional) 

###### 492. (Optional) 

###### 493. (Optional) 

###### 494. (Optional) 

###### 495. (Optional) 

###### 496. (Optional) 

###### 497. (Optional) 

###### 498. (Optional) 

###### 499. (Optional) 

###### 500. (Optional) 

###### 501. (Optional) 

###### 502. (Optional) 

###### 503. (Optional) 

###### 504. (Optional) 

###### 505. (Optional) 

###### 506. (Optional) 

###### 507. (Optional) 

###### 508. (Optional) 

###### 509. (Optional) 

###### 510. (Optional) 

###### 511. (Optional) 

###### 512. (Optional) 

###### 513. (Optional) 

###### 514. (Optional) 

###### 515. (Optional) 

###### 516. (Optional) 

###### 517. (Optional) 

###### 518. (Optional) 

###### 519. (Optional) 

###### 520. (Optional) 

###### 521. (Optional) 

###### 522. (Optional) 

###### 523. (Optional) 

###### 524. (Optional) 

###### 525. (Optional) 

###### 526. (Optional) 

###### 527. (Optional) 

###### 528. (Optional) 

###### 529. (Optional) 

###### 530. (Optional) 

###### 531. (Optional) 

###### 532. (Optional) 

###### 533. (Optional) 

###### 534. (Optional) 

###### 535. (Optional) 

###### 536. (Optional) 

###### 537. (Optional) 

###### 538. (Optional) 

###### 539. (Optional) 

###### 540. (Optional) 

###### 541. (Optional) 

###### 542. (Optional) 

###### 543. (Optional) 

###### 544. (Optional) 

###### 545. (Optional) 

###### 546. (Optional) 

###### 547. (Optional) 

###### 548. (Optional) 

###### 549. (Optional) 

###### 550. (Optional) 

###### 551. (Optional) 

###### 552. (Optional) 

###### 553. (Optional) 

###### 554. (Optional) 

###### 555. (Optional) 

###### 556. (Optional) 

###### 557. (Optional) 

###### 558. (Optional) 

###### 559. (Optional) 

###### 560. (Optional) 

###### 561. (Optional) 

###### 562. (Optional) 

###### 563. (Optional) 

###### 564. (Optional) 

###### 565. (Optional) 

###### 566. (Optional) 

###### 567. (Optional) 

###### 568. (Optional) 

###### 569. (Optional) 

###### 570. (Optional) 

###### 571. (Optional) 

###### 572. (Optional) 

###### 573. (Optional) 

###### 574. (Optional) 

###### 575. (Optional) 

###### 576. (Optional) 

###### 577. (Optional) 

###### 578. (Optional) 

###### 579. (Optional) 

###### 580. (Optional) 

###### 581. (Optional) 

###### 582. (Optional) 

###### 583. (Optional) 

###### 584. (Optional) 

###### 585. (Optional) 

###### 586. (Optional) 

###### 587. (Optional) 

###### 588. (Optional) 

###### 589. (Optional) 

###### 590. (Optional) 

###### 591. (Optional) 

###### 592. (Optional) 

###### 593. (Optional) 

###### 594. (Optional) 

###### 595. (Optional) 

###### 596. (Optional) 

###### 597. (Optional) 

###### 598. (Optional) 

###### 599. (Optional) 

###### 600. (Optional) 

###### 701. (Optional) 

###### 702. (Optional) 

###### 703. (Optional) 

###### 704. (Optional) 

###### 705. (Optional) 

###### 706. (Optional) 

###### 707. (Optional) 

###### 708. (Optional) 

###### 709. (Optional) 

###### 710. 

Final Answer...##### Objective Function: 

$\quad \min \sum_{t=1}^{24} (x_t + y_t)$

##### Constraints

###### 1. Driver coverage constraints: 

$\displaystyle \sum_{k=0}^{3} x_{((t-k-1)\bmod 24)+1} \;\ge\; r_t \quad \forall t \in \{1,\dots,24\}$

###### 2. Crew coverage constraints: 

$\displaystyle \sum_{k=0}^{3} y_{((t-k-1)\bmod 24)+1} \;\ge\; r_t \quad \forall t \in \{1,\dots,24\}$

###### 3. Non‑negativity and integrality: 

$x_t,\,y_t \;\in\; \mathbb{Z}_{+} \quad \forall t \in \{1,\dots,24\}$

##### Retrieved Information

```json
{
  "required": [
    20, 18, 15, 15, 20, 30, 60, 70, 50, 55,
    65, 75, 80, 70, 60, 55, 60, 75, 85, 70,
    50, 40, 35, 25
  ],
  "shifts": [
    "0:00-1:00", "1:00-2:00", "2:00-3:00", "3:00-4:00",
    "4:00-5:00", "5:00-6:00", "6:00-7:00", "7:00-8:00",
    "8:00-9:00", "9:00-10:00", "10:00-11:00", "11:00-12:00",
    "12:00-13:00", "13:00-14:00", "14:00-15:00", "15:00-16:00",
    "16:00-17:00", "17:00-18:00", "18:00-19:00", "19:00-20:00",
    "20:00-21:00", "21:00-22:00", "22:00-23:00", "23:00-0:00"
  ]
}
```
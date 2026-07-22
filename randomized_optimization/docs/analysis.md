# Randomized Optimization

> Portfolio analysis notes converted from prior report drafts. Author bylines and course identifiers removed.

Randomized Optimization
				
Introduction
Randomized optimization (RO) is the process of finding the min or max of problems without requiring the problem’s gradient to be optimized. Thus, RO can be used on functions that are not continuous or differentiable. It operates by iteratively moving to better locations in the search space.
This analysis compares four randomized optimization algorithms – random hill climbing (RHC), simulated annealing (SA), genetic algorithms (GA), and mutual information maximizing input clustering (MIMIC) via the use of three optimization problems and a neural network. The problems were selected to highlight the weaknesses and strengths of each RO algorithm.
This project utilizes scikit-learn and mlrose to implement and analyze these randomized optimization and neural net problems. Utilizing MLRose Runners, various restarts (RHC), population sizes and mutation rates (GA), temperatures (SA), and keep percent and population sizes (MIMIC) were tested to find ideal parameters before creating the fitness curves.
Algorithms
RHC starts each time with an indiscriminate guess and then moves towards increased fitness on each iteration. Thus, it is optimal for convex solution spaces. RHC uses relatively little computer memory.
SA is a non-deterministic search, hill climbing algorithm. SA occasionally accepts a decreased fitness function to get away from a local optimum. It’s temperature parameter determines the possibility of accepting an inferior solution where the higher the temperature, the higher probability of accepting. Typically, the temperature decreases as SA runs. Theoretically SA can find the global optimum but could be very slow. Lastly, it can be difficult to determine the degree at which to decrease the temperature.
GA is an evolutionary algorithm that is a metaheuristic method influenced by natural selection. It allows crossover and mutation on the two parents and places the offspring in a new population from which it returns the best solution. GA doesn’t need to know the fitness function derivative and can arbitrarily explore the population and find solutions that local search algorithms cannot. Nonetheless, this technique may not be ideal for complex questions.
MIMIC tries to exploit the problem’s underlying structure to eliminate re-exploration of sub-optimal solution space segments in future iterations. It does this by first arbitrarily sampling from regions of the input space most probable to include the optima and then using a density estimator to process a variety of structure on the input space.
FlipFlop (SA)
Flipflop counts the number of bit flips in an array of 0s and 1s with the max of n-1 flips. In other words, the max fitness bit string would consist solely of the alternating digits. This specific problem utilized 50 bits. The RHC ideal restart was 0. The GA ideal population size was 10 and mutation rate was 0.4. The SA ideal temp was 10; and the MIMIC ideal keep percent was 0.5 and population size was 500.
As the left graph above and summarization table below shows SA and MIMIC achieved the best (highest) level of fitness. However, SA (6.5 ms) achieved it much faster than MIMIC (489.32 ms). This was due to the simplicity of the problem space – there was not really anything useful or necessary to learn about the underlying distribution so the additional work that MIMIC does to reduce the H space in future iterations was largely unnecessary. Additionally, it is interesting how different each fitness curve is. This is likely due to the inherent differences between the algorithms.
Flipflop was designed to be an exploitation scenario. SA and RHC are a simple algorithm that is most successful when exploitation of neighbor knowledge in a problem space leads to optimal solution.
SA also excels when the fitness function f(x) is computationally inexpensive. It also has a higher chance of finding the global maximum when there is a large basin of attraction that will inevitably lead to the max. These properties are highlighted by flipflop which has a single max with a small but increasing slope toward the solution.
In future tests, smaller ideal temperatures for SA can be tried since 10 was chosen by the runner but was also the smallest option that the runner was given.

MIMIC
RHC
GA
SA
Fitness at Best State
49
37
48
49
Run Time [ms]
489.32
2.1
115.99
6.5

KnapSack (MIMIC)
Knapsack is a combinatorial, resource allocation, optimization question that asks: given a group of items with a value and weight, find the count of each item to include so the value is as large as possible and the total weight is less than or equal to a given limit. This specific instance used 35 items types and 20 max items. In theory, the search space of this problem is much smaller, but the limit of item weights will make this problem NP hard. The RHC ideal restart was 0. The GA ideal population size was 500 and mutation rate was 0.2. The SA ideal temp was 1000; and the MIMIC ideal keep percent was 0.5 and population size was 500.
While MIMIC came in second in this instance for fitness, its theoretically more tuned for this type of problem because of the problems smaller search space and because it is easier to find the dependency relation of each item and thus estimate a probability distribution using a dependency tree. The slight better fitness from GA is likely due to better parameter selection. It is interesting that GA’s ideal population size and MIMIC’s were the same for knapsack but were not for flipflop. This is likely coincidental, but in future efforts, it be best to try higher keep percent and population sizes since MIMIC maxed on those parameter options.
MIMIC attempts to learn about the underlying distribution with each iteration. As N increases and the distribution of item subsets become larger, it makes sense that MIMIC would take longer to return its optimal value. The knowledge MIMIC gathers allows it to return strictly better solutions until it eventually converges on a global optimum or runs out of iteration time and returns the best it has found so far. 
Since the knapsack solution space is large with many small basins of attraction for local optima, RHC, SA, and GA have a higher likelihood of getting trapped in the local optima and never exploring the area that has the global optimum. Additionally, instead of randomly bouncing around between selections of items (like RHC, SA, and GA), it’s optimal for the solver to learn something about the underlying distribution of possible solutions after each round, to minimize the number of f(x) evaluations in future iterations and prevent re-exploring known bad solutions.
Lastly, there is theoretically no underlying locality in the solutions of this problem space that GA can exploit either, particularly with single point crossover. MIMIC reached its max fitness in fewer iterations than GA.

MIMIC
RHC
GA
SA
Fitness at Best State
1747
1450

1849
1627
Run Time [ms]
1689.38
0.451
636.46
1.528

Traveling Salesman (GA)
TSP is a graph theory problem to find an optimal traversal route with minimum weights sum. The question it asks is: find the shortest route that visits a batch of cities and returns to the beginning.
This is NP-hard and this specific instance used 22 cities. The RHC ideal restart was 0. The GA ideal population size was 500 and mutation rate was 0.2. The SA ideal temp was 100; and the MIMIC ideal keep percent was 0.25 and population size was 100. Since GA’s ideal population size was at the max provided option of 500, future runs can test if increasing the options past 500 increases the ideal population size.
While in this test run, MIMIC had the highest fitness; it took an unreasonable amount of time to run. This is because MIMIC uses joint probabilities to develop an understanding of dependencies but the exact dependency between alternative routes cannot be driven easily due to the large search space of directed graphs.
The next highest was RHC which also had the lowest run time. However, as with all these algorithms, a lot of their performance is determined by the chosen parameters’ values. It is interesting that RHC quickly randomized into an optimum.
GA while it wasn’t the highest, isn’t too far from RHC and SA on fitness, and theoretically is optimal than the other algorithms for this type of problem. This is because the cost of a traveling route is related to the order of places it visited and the mating and mutation of GA can help optimize many small local routes and thus is more likely to get the global optimal result. GA excels when f(x) is quick to compute and when it can exploit some level of locality within the problem structure.

MIMIC
RHC
GA
SA
Fitness at Best State
853.99
481.14
450.04
473.83
Run Time [ms]
10,959.55
2.45
1281.59
4

Neural Net
Figure 1: Gradient Descent
Figure 1: Gradient Descent
Figure 3: Genetic Algorithm
Figure 3: Genetic Algorithm
Figure 2: RHC
Figure 2: RHC
A first interesting observation is the difference among the algorithms between the prediction distribution over the classes. RHC and SA had flipped prediction distributions, i.e., more malignant than benign. GA had a similar proportion to the dataset. This is likely due to a lack of consideration of the underlying structure by RHC and SA and is also reflected in the low precision and accuracy rates.
Backpropagation with gradient descent was implemented, and it achieved similar results as the similar setup from a previous analysis (results reported in the below table). RO techniques were not expected to perform better than backpropagation when training a neural net and they did not in reality. One reason is because Ros like SA and RHC were expected to fall into local optima.
Figure 1: Loss vs Iterations (GD is *-1)
Figure 1: Loss vs Iterations (GD is *-1)
The graph shows that as the iterations increased, RHC and SA loss slightly decreased but was still relatively bad compared to GA and GD. Gradient descents loss progressively got better with more iterations (the above graph plots gradient descent’s loss * -1).
Lastly, while the prediction times were quick for all, the training time was poor for the RO algorithms. Although GA received a much closer F1 Score to gradient descent, it required drastically more training time. Future tests using the neural network can try more parameter variations and alternative randomized optimization algorithms; however, it is unlikely that it will result in an overall performance as good as gradient descent.

Gradient Descent
RHC
GA
SA
Recall
0.875
0.900
0.825
0.925
Precision
1.000
0.330
0.971
0.336
Accuracy
0.956
0.325
0.930
0.333
F1 Score
0.933
0.483
0.892
0.493
Training Time [ms]
7.5583
410.8338
1470.4102
8.0906
Prediction Time [ms]
0.0001
0.0001
0.0001
0.0001

Conclusion
The experiments were designed to highlight the weaknesses and strengths of SA, RHC, GA, and MIMIC. Yet, a key item it did highlight was the importance of parameter selection. Widely different results can be achieved for the same problem statements if parameters are different. The results also made it apparent that good knowledge of the problem space is critical when choosing which RO algorithm to use to find an optimal solution. In general, if the underlying solution set distribution has no structure, SA and RHC can be good since a random walk is about as good as you are able to get. If there is some structure, then MIMIC could be better so you don’t waste time continually exploring bad areas of the problem space in future iterations. Additionally, if there is some degree of locality between attributes in your solution space, the evolution-like crossover of GA might get you the optimal solution.
The analysis also highlights the importance of time. There are many conflicting priorities in life and accuracy and completion time are two common ones. Sometimes it may be more important to achieve a higher fitness at the expense of time and other times a decent fitness but quicker time is better.
The optimization problems above also showed that RHC has the advantages of being fast and easy with no parameter tuning but is not suitable for too complicated problems and can be hard to converge. SA can be fast and good at approximating global optima but needs parameter tuning and can sometimes run slow. Different cooling temperatures for SA can have a huge impact on performance. GA is sometimes fast and performs well for complicated problems, but needs parameter tuning, can run very slow on complicated problems, and has high volatility to converge. Lastly, MIMIC is suitable for complicated problems and can be good at approximating global optima but needs parameter tuning and is very slow.
Regarding the neural net, the data collected showed that backpropagation was a better method for optimizing the weights of a neural network than the RO algorithms are in terms of accuracy and training time.
References
Lbronchal. (2017, November 21). Breast cancer dataset analysis. Retrieved February 19, 2021, from https://www.kaggle.com/lbronchal/breast-cancer-dataset-analysis
Knapsack problem. (2021, March 06). Retrieved March 15, 2021, from https://en.wikipedia.org/wiki/Knapsack_problem
Travelling salesman problem. (2021, February 14). Retrieved March 15, 2021, from https://en.wikipedia.org/wiki/Travelling_salesman_problem

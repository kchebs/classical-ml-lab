# Sequential Decisions (MDPs & Q-Learning)

> Portfolio analysis notes converted from prior report drafts. Author bylines and course identifiers removed.

Proj. 4: Markov Decision Processes & Reinforcement Learning
				
Introduction
This project explores various techniques an agent can use to make decisions in Markov decision processes (MDP). The two MDP problems - Forest and Frozen Lake - are analyzed using value iteration, policy iteration, and the model-free reinforcement learning algorithm Q-Learning. Forest and Frozen Lake were chosen due to their interestingness and to demonstrate the different behaviors with large and small number of states. 
MDPs are discrete-time stochastic control processes often defined as (S, A, Pa, Ra) or (S, A, T, R, gamma) where S is the set of all possible states, A is a fixed set of actions, Pa or T is the probability to go from one state to another for a given action, R is the reward received after transitioning states due to action A, and gamma is the discount factor which reduces the importance of future rewards thereby affecting how the MDP will plan. The goal in a MDP is to find a good “policy” for the decision maker/maximize reward and they are particularly useful in situations that require large sequences of actions where the decisions are better made using expected values and the actions are not entirely reliable.
Techniques
This project utilizes three techniques. Policy iteration creates a policy and evaluates the expected value at all states under that policy. It then explores whether the value at a certain state can increase by modifying the policy/changing the action, thereby iterating until there is no improvement after which it continues through the policy. The technique eventually converges by solving its set of linear equations when determining the expect value of all states. 
Value iteration applies the Bellman equation to evaluate progressively state-by-state until the solution converges. The utility is not initially known but the utility of the immediate reward is. The utility of the nearest states is calculated based on discounted reward using the reward at the terminal state and this iterates until the utility of all states are known. Policy and value iteration use domain knowledge as the foundation of the transition function. The policy and value iterations were implemented in Python using MDPToolbox.
Q-learning works by assigning all states a value then visiting each state to reassess the value based on immediate and delayed rewards. In early states, Q-learning will explore more and as it progresses it will begin to exploit more. Convergence occurs if it reaches a plateauing optimal reward or, in other words, if multiple episodes achieve similar policy rewards.
Forest
The forest problem is an interesting forest management dilemma to wait and maintain a forest or to cut the trees. Although, this is called forest, it can be also helpful for communities maintaining hiking trails through green areas, businesses trying to make money by selling wood, and parks. P was set to 0.001, reward1 to 100, reward2 to 10, and the number of states is 1000 except in the experiments where it was varied.
For the below graphs, these experiments were run with a gamma/discount factor of 0.99 and for value iteration, epsilon was 0.001. For Q-learing, the alpha schedule was 0.01 * the number of episodes and the epsilon schedule was 0.5 * number of episodes. The Q-Learning location starts at a random location. The differences between the three techniques are most prevelant in the max V graph where each technique had a different shape. Max V is the max optimal value function at each state. Varying the number of states impacted the mean and max V. The mean V decreased with higher number of states for both policy and value iterations. As expected, the number of iterations to converge and the time to converged increased when the number of states also increased. However, the iterations to converge plateued.
For both iteration techniques, the policy matched regardless of varying the number of states and gamma as shown in the left graph. As discount rate increased the number of iterations to converge, max V, mean V, and optimal policy reward all increased for the policy iteration as seen in the right graph. The increase in value and rewards is expected as a discount rate closer to 1 means that rewards that are far in the future will be almost as important as immediate rewards. In other words, discount rates impact how we estimate the value of each state.
Lastly, when comparing time, policy iteration was the slowest. Value iteration also achieved a minimum error faster than policy iteration. Q-Learning was the fastest. This is as expected because Q-learning does the least computation each step and policy iteration does the most. To understand how problem size affects running time of the two algorithms, its also reasonable to use their computational complexity. The policy iteration and value iteration efficiences are O(S^2N) and O(S^2A) respectively where S is the number of states, A is the number of possible actions at each state and N is the number of policy evaluations in each iteration. Large state spaces problems such as this one can suffer because even a small increase in the number of iterations can result in large increases in time required to converge. 
For the main section of this problem and for comparison of the techniques, the Q-Learning was conducted with 100,000 episodes. The impact of the number of episodes is shown in the below graph.
Episodes are simply each iteration made by an agent. Thus, it was expected that the time, mean V, and max V would increase as episode increased. For this problem, value iteration delivered negligbly more value than policy iterations but policy iterations converged in less iterations. Q-Learning did not perform well compared to policy and value iteration. While Q-Learning was fast, it did not deliver anywhere near the value and policy that the other two techniques did. The difference between value and policy iteration was 0, thus they converge to the same answer; but their difference with Q-Learning was 715,123 value and 460 policy.

Policy Iteration
Value Iteration
Q-Learning
Runtime per Iteration or Episode
0.306 sec
0.000608 sec
0.000256 sec
Iterations to Converge
479
1458
Did Not Converge

Frozen Lake
The Frozen Lake problem is a grid world problem with 8x8 dimensions. This type of problem is interesting because its common in many video and board games and could even be extrapolated to hide-and-seek. The state space is 64 because of its dimensions. There are 4 discrete actions and a -0.1 penalty for failing into a hole. It was implemented using the Gym library in Python.

Like with the forest problem and exemplified in the maps below, policy and value iteration were more like each other than they were to Q-Learning. The Q-Learning was conducted with 10,000 episodes and a 0.99 gamma and its location starts at 0.

In the above graphs, the yellow square represents the start space. The red square are the holes in the ice, which an agent would be penalized for entering; and the green space is the terminal space where the reward is +1.

When changing gamma, the policy did occasionally diverge between the value and policy iteration techniques. This was not initially expected because it did not happen for the forest problem but is understandable due to how policy and value iterations vary. Additionally, forest value iteration achieved a minimum error first, but for frozen lake, policy iteration achieved the lowest error first. This is also understandable given that policy finds a value function then finds an improved policy based on the new value function and iterates.
As expected, more episodes led to higher time and mean V, which also occurred in the forest problem. However, the sudden drop in max V at a low episode was unexpected but likely is due to the randomness of Q-Learning.

Unlike the forest problem, Q-Learning had the highest runtime per episode. However, like forest, policy iteration runtime per iteration was higher than value iterations’. In both problems, while policy iteration took longer per an iteration, it took fewer iterations. This is expected because policy iteration evaluates the utility of each state-action pair for a given policy which takes more time but provides more information. The difference between value and policy iteration was 0 and between Q-Learning was 4.24 value and 15 policy. Unlike in the forest problem, the Q-Learning algorithm converge for this small state space algorithm as highlighted in the above plateauing mean V.

Policy Iteration
Value Iteration
Q-Learning
Runtime per Iteration or Episode
0.00419 sec
0.000114 sec
0.0126 sec
Iterations to Converge
10
296
6000

Conclusion
From analyzing the results, Q-Learning has the advantages of being quick because of its randomness but may not converge. This is impacted by the choice of epsilon, a threshold for random generation below which an agent chooses to explore and the probability an action will be random. This determines the “exploration versus exploitation” of the algorithm. Lastly, Q-learnings runtime is nearly linear to the number of iterations and works when prior knowledge is nonexistent.
Policy and value iterations have many similarities and in the problems above converged to the same answer. Yet, policy iteration includes a policy evaluation phase plus policy improvement phase with repeats while value iteration includes finding optimal value function plus one policy extraction phase. These slight differences translate into policy iterations often converging in fewer iterations, but the value iteration technique having a smaller runtime per an iteration.
Thus, the best technique is often problem specific and depends on what you know/if anything about the domain. This leads to some possible future work involving investigating these techniques on problems of other domains, investigating problems that have states other than the terminal state that are beneficial to the agent, and problems with circular routes that may trap an algorithm. Afterwards, one can also investigate other options instead of using a Q-Learning technique.

# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        k = 0
        while k < self.iterations:
            newValues = self.values.copy()
            for state in self.mdp.getStates():
                #update value of each state
                if not self.mdp.isTerminal(state):
                    possibleActions = self.mdp.getPossibleActions(state)
                    maxExpectimaxValue = self.computeQValueFromValues(state, possibleActions[0])
                    for action in possibleActions:
                        expectimaxValue = self.computeQValueFromValues(state, action)
                        if expectimaxValue > maxExpectimaxValue:
                            maxExpectimaxValue = expectimaxValue
                    newValues[state] = maxExpectimaxValue
                    
            self.values.update(newValues)
            k += 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        sum = 0
        for transitionState in self.mdp.getTransitionStatesAndProbs(state, action):
            newState = transitionState[0]
            probability = transitionState[1]
            reward = self.mdp.getReward(state, action, newState)
            val = probability * (reward + (self.discount * self.values[newState]))
            sum += probability * (reward + (self.discount * self.values[newState]))
        return sum

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return
        possibleActions = self.mdp.getPossibleActions(state)
        actionTaken = possibleActions[0]
        maxExpectimaxValue = self.computeQValueFromValues(state, possibleActions[0])
        for action in possibleActions:
            expectimaxValue = self.computeQValueFromValues(state, action)
            if expectimaxValue > maxExpectimaxValue:
                maxExpectimaxValue = expectimaxValue
                actionTaken = action
        return actionTaken

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        k = 0
        while k < self.iterations:
            statesList = self.mdp.getStates()
            stateIndex = k % len(statesList)
            state = statesList[stateIndex]

            if not self.mdp.isTerminal(state):
                possibleActions = self.mdp.getPossibleActions(state)
                maxExpectimaxValue = self.computeQValueFromValues(state, possibleActions[0])
                for action in possibleActions:
                    expectimaxValue = self.computeQValueFromValues(state, action)
                    if expectimaxValue > maxExpectimaxValue:
                        maxExpectimaxValue = expectimaxValue
                self.values[state] = maxExpectimaxValue
                    
            k += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        #initialize sets
        for state in self.mdp.getStates():
            predecessors[state] = set()
        #compute predecessors
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for transitionState in self.mdp.getTransitionStatesAndProbs(state, action):
                    newState = transitionState[0]
                    probability = transitionState[1]
                    if probability > 0:
                        predecessors[newState].update({state})
        #initialize empty priority queue
        pQueue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                currentVal = self.values[state]

                #Should probably change this maxQValue block into a seperate function but I don't really care
                possibleActions = self.mdp.getPossibleActions(state)
                maxExpectimaxValue = self.computeQValueFromValues(state, possibleActions[0])
                for action in possibleActions:
                    expectimaxValue = self.computeQValueFromValues(state, action)
                    if expectimaxValue > maxExpectimaxValue:
                        maxExpectimaxValue = expectimaxValue

                diff = abs(currentVal - maxExpectimaxValue)

                pQueue.push(state, (0 - diff))
        k = 0
        while k < self.iterations:
            if pQueue.isEmpty():
                return
            poppedState = pQueue.pop()
            if not self.mdp.isTerminal(poppedState):
                #update state value
                possibleActions = self.mdp.getPossibleActions(poppedState)
                maxExpectimaxValue = self.computeQValueFromValues(poppedState, possibleActions[0])
                for action in possibleActions:
                    expectimaxValue = self.computeQValueFromValues(poppedState, action)
                    if expectimaxValue > maxExpectimaxValue:
                        maxExpectimaxValue = expectimaxValue
                self.values[poppedState] = maxExpectimaxValue
            for p in predecessors[poppedState]:
                possibleActions = self.mdp.getPossibleActions(p)
                maxExpectimaxValue = self.computeQValueFromValues(p, possibleActions[0])
                for action in possibleActions:
                    expectimaxValue = self.computeQValueFromValues(p, action)
                    if expectimaxValue > maxExpectimaxValue:
                        maxExpectimaxValue = expectimaxValue
                diff = abs(self.values[p] - maxExpectimaxValue)
                if diff > self.theta:
                    pQueue.update(p, (0 - diff))
            k += 1




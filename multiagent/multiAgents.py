# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best


        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        ghost_factor = -5
        food_factor = 5
        greedy_factor = 5

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        ghost_distances = [manhattanDistance(newPos, ghost_state.getPosition()) for ghost_state in newGhostStates]
        ghost_weight = ghost_factor*get_avg_dist(ghost_distances, 1000)

        food_distances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        food_weight = food_factor*get_avg_dist(food_distances, 0.1)

        greedy_food_weight = greedy_factor*get_min_dist(food_distances)

        score_change = successorGameState.data.score - currentGameState.data.score
        
        return score_change + ghost_weight + food_weight + greedy_food_weight
        

def get_min_dist(l):
    # helper function
    try:
      return 1 / float(min(l))
    except ValueError:
      return 999

def get_avg_dist(l, default_value):
    # helper function
    if len(l) == 0:
      return 1000
    return sum([1/float(i) if i != 0 else default_value for i in l])/len(l)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        actions = gameState.getLegalActions(0)
        best_score = -9999999
        best_action = None
        for action in actions:
          score = self.minimax(gameState.generateSuccessor(0, action), 0, 1)
          if score > best_score:
            best_score = score
            best_action = action
        return best_action

    def is_terminal_state(self, state, actions, depth):
        return len(actions) == 0 or state.isWin() or state.isLose() or depth == self.depth

    def minimax(self, state, depth, agent_index):

      if agent_index == state.getNumAgents():
          depth += 1
          agent_index = 0
      
      actions = state.getLegalActions(agent_index)
      
      if self.is_terminal_state(state, actions, depth):
        return self.evaluationFunction(state)

      # maximiser
      if agent_index == 0:
        best_value = -99999999
        for action in actions:
          v = self.minimax(state.generateSuccessor(agent_index, action), depth, agent_index + 1)
          best_value = max(best_value, v)
        return best_value

      #minimiser
      else:
        best_value = 99999999
        for action in actions:
          v = self.minimax(state.generateSuccessor(agent_index, action), depth, agent_index + 1)
          best_value = min(best_value, v)
        return best_value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        actions = gameState.getLegalActions(0)
        best_score = -9999999
        best_action = None
        alpha = -9999999
        beta = 9999999
        for action in actions:
          score = self.minimax(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
          if score > best_score:
            best_score = score
            best_action = action
            alpha = best_score
        return best_action

    def is_terminal_state(self, state, actions, depth):
        return len(actions) == 0 or state.isWin() or state.isLose() or depth == self.depth

    def minimax(self, state, depth, agent_index, alpha, beta):

      if agent_index == state.getNumAgents():
          depth += 1
          agent_index = 0
      
      actions = state.getLegalActions(agent_index)
      if self.is_terminal_state(state, actions, depth):
        return self.evaluationFunction(state)

      if agent_index == 0:
        best_value = -99999999
        for action in actions:
          v = self.minimax(state.generateSuccessor(agent_index, action), depth, agent_index + 1, alpha, beta)
          best_value = max(best_value, v)
          alpha = max(alpha, best_value)
          if beta < alpha:
            break
        return best_value

      else:
        best_value = 99999999
        for action in actions:
          v = self.minimax(state.generateSuccessor(agent_index, action), depth, agent_index + 1, alpha, beta)
          best_value = min(best_value, v)
          beta = min(beta, best_value)
          if beta < alpha:
            break
        return best_value

class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        actions = gameState.getLegalActions(0)
        best_score = -9999999
        best_action = None
        for action in actions:
          score = self.minimax(gameState.generateSuccessor(0, action), 0, 1)
          if score > best_score:
            best_score = score
            best_action = action
        return best_action

    def is_terminal_state(self, state, actions, depth):
        return len(actions) == 0 or state.isWin() or state.isLose() or depth == self.depth

    def minimax(self, state, depth, agent_index):


      if agent_index == state.getNumAgents():
          depth += 1
          agent_index = 0
      
      actions = state.getLegalActions(agent_index)
      if self.is_terminal_state(state, actions, depth):
        return self.evaluationFunction(state)

      if agent_index == 0:
        best_value = -99999999
        for action in actions:
          v = self.minimax(state.generateSuccessor(agent_index, action), depth, agent_index + 1)
          best_value = max(best_value, v)
        return best_value

      else:
        best_value = 99999999
        values = [self.minimax(state.generateSuccessor(agent_index, action), depth, agent_index + 1) for action in actions]
        return sum(values)/float(len(values))

def betterEvaluationFunction(game_state):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    ghost_factor = -5
    food_factor = 10

    pacman_position = game_state.getPacmanPosition()
    food_grid = game_state.getFood()
    ghost_states = game_state.getGhostStates()
    scared_times = [ghost_state.scaredTimer for ghost_state in ghost_states]

    ghost_distances = [manhattanDistance(pacman_position, ghost_state.getPosition()) for ghost_state in ghost_states]

    for i, st in enumerate(scared_times):
      if ghost_distances[i] < st:
          ghost_factor = 100

    ghost_weight = ghost_factor*get_avg_dist(ghost_distances, 1000)

    food_distances = [manhattanDistance(pacman_position, food) for food in food_grid.asList()]
    food_weight = food_factor*get_avg_dist(food_distances, 0.1)

    game_score = game_state.getScore()

    return ghost_weight + food_weight + game_score

# Abbreviation
better = betterEvaluationFunction

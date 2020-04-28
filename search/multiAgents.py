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
##        print("scores:" + str(scores))
##        print(legalMoves)
        "Add more of your code here if you want to"

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        currFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        power = currentGameState.getCapsules()
        nFood=newFood.asList()
        food= currFood.asList()
        
        "*** YOUR CODE HERE ***"
        closestFood = 999999999
        if len(nFood)==0:
            closestFood = 0
            
        for f in nFood:
            temp = manhattanDistance(newPos, f)
            if temp < closestFood:
                closestFood = temp
                x = f        
            
        score = float(1)/(float(closestFood)+1)+successorGameState.getScore()
##        score = 1/((closestFood)+1)+successorGameState.getScore()
        cPellet = 9999999999
        if len(power)> 0:
            for pellet in power:
                temp = manhattanDistance(newPos, pellet)
                if temp < cPellet:
                    cPellet = temp
                    y = pellet
            score = float(1)/(float(cPellet)+1) + successorGameState.getScore()
        

        
        ghostList = []
        closestGhost = 9999999999
        for i in range(len(newGhostStates)):
            ghostList.append(successorGameState.getGhostPosition(i+1))
            ghost_dis = manhattanDistance(newPos, successorGameState.getGhostPosition(i+1))
            if ghost_dis<closestGhost:
                closestGhost = ghost_dis
                index = i+1

        distanceFromGhost = manhattanDistance(newPos, successorGameState.getGhostPosition(index))
        if distanceFromGhost <= 1:
            score = -999999

        
        
            
        return score

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
        "*** YOUR CODE HERE ***"

        def helper(gameState, agent, count):
            agents = gameState.getNumAgents()
            if agent>agents-1:
                agent = 0
                count+=1
            
                
            inf = float('inf')
            best_move = 'Stop'
            
            if (count == self.depth) or gameState.isWin() or gameState.isLose():
                return best_move, self.evaluationFunction(gameState)
            if agent == 0:
                value = -inf
            else:
                value = inf
            for action in gameState.getLegalActions(agent):
                nxt_pos = gameState.generateSuccessor(agent, action)
                nxt_move, nxt_val = helper(nxt_pos, agent + 1, count)
                if agent == 0 and value < nxt_val:
                    value, best_move = nxt_val, action
                if agent > 0 and value > nxt_val:
                    value, best_move = nxt_val, action
            return best_move, value
        best_move, value = helper(gameState, 0, 0)
        return best_move
 

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        inf = float('inf')
        def abhelper(gameState, agent, count, alpha, beta):
            best_move = 'Stop'
            
            agents = gameState.getNumAgents()

            if agent==agents:
                agent = 0
                count += 1

            if count == self.depth or gameState.isWin() or gameState.isLose():
                return best_move, self.evaluationFunction(gameState)

            if agent == 0:
                value = -inf
            else:
                value = inf

            for action in gameState.getLegalActions(agent):
                nxt_pos = gameState.generateSuccessor(agent, action)
                nxt_move, nxt_val = abhelper(nxt_pos, agent + 1, count, alpha, beta)
                if agent == 0:
                    if value < nxt_val:
                        value, best_move = nxt_val, action
                    if value >= beta:
                        return best_move, value
                    alpha =max(alpha, value)
                else:
                    if value > nxt_val:
                        value, best_move = nxt_val, action
                    if value <= alpha:
                        return best_move, value

                    beta = min(beta, value)
            return best_move, value

        best_move, value = abhelper(gameState, 0, 0, (-inf), inf)
        return best_move
    
                    
                

            
            
            
            

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def exhelper(gameState, agent, count):
            agents = gameState.getNumAgents()
            if agent>agents-1:
                agent = 0
                count+=1
            
                
            inf = float('inf')
            best_move = 'Stop'
            
            if (count == self.depth) or gameState.isWin() or gameState.isLose():
                return best_move, self.evaluationFunction(gameState)
            if agent == 0:
                value = -inf
            else:
                value = 0
            moves = gameState.getLegalActions(agent)
            for action in moves:
                nxt_pos = gameState.generateSuccessor(agent, action)
                nxt_move, nxt_val = exhelper(nxt_pos, agent + 1, count)
                if agent == 0 and value < nxt_val:
                    value, best_move = nxt_val, action
                if agent > 0:
                    value = value + (float(1)/float(len(moves))) * nxt_val
            return best_move, value

        best_move, value = exhelper(gameState, 0, 0)
        return best_move
        
        
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    currFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    power = currentGameState.getCapsules()
##    nFood=newFood.asList()
    food= currFood.asList()

    "*** YOUR CODE HERE ***"
    closestFood = 999999999
    if len(food)==0:
        closestFood = 0

    for f in food:
        temp = mazeDistance(newPos, f, currentGameState)
        if temp < closestFood:
            closestFood = temp
            x = f        

    score = -float(1)/(float(closestFood)+1)+currentGameState.getScore()
    ##        score = 1/((closestFood)+1)+successorGameState.getScore()
    cPellet = 9999999999
    if len(power)> 0:
        for pellet in power:
            temp = manhattanDistance(newPos, pellet)
            if temp < cPellet:
                cPellet = temp
                y = pellet
        #score += -float(1)/(float(cPellet)+1) + currentGameState.getScore()

    ghostList = []
    closestGhost = 9999999999
    for i in range(len(newGhostStates)):
        ghostList.append(currentGameState.getGhostPosition(i+1))
        ghost_dis = manhattanDistance(newPos, currentGameState.getGhostPosition(i+1))
        if ghost_dis<closestGhost:
            closestGhost = ghost_dis
            index = i+1

    distanceFromGhost = manhattanDistance(newPos, currentGameState.getGhostPosition(index))
    if distanceFromGhost <= 1:
        score = -999999
##    score += distanceFromGhost
        
    return score

##def mazeDistance(point1, point2, gameState):
##    """
##    Returns the maze distance between any two points, using the search functions
##    you have already built. The gameState can be any game state -- Pacman's
##    position in that state is ignored.
##
##    Example usage: mazeDistance( (2,4), (5,6), gameState)
##
##    This might be a useful helper function for your ApproximateSearchAgent.
##    """
##    x1, y1 = point1
##    x2, y2 = point2
##    walls = gameState.getWalls()
##    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
##    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
##    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
##    return len(search.bfs(prob))
##
##def bfs(problem):
##    """Search the shallowest nodes in the search tree first."""
##    "*** YOUR CODE HERE ***"
##    start = (problem.getStartState())
##    q = util.Queue()
##    q.push((start, []))
##    visited = []
##    
##    while not q.isEmpty():
##        
##        state, n = q.pop()
##        if (not (state in visited)):
##            visited.append(state)
##            if (problem.isGoalState(state)):
##                return n
##            else:
##                for succ in problem.getSuccessors(state):
##                    q.push((succ[0], n + [succ[1]]))               
##
##    return False


    
    

# Abbreviation
better = betterEvaluationFunction


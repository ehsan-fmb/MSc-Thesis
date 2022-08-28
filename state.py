import sys

class State:
    def __init__(self,game,parent=None,ai=None,Q=0):
        self.game=game
        self.parent = parent
        self.q=Q
        self.action_index=ai
        self.n=sys.float_info.epsilon
        self.children=[]
        self.subgoals=[]
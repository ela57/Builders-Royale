"""Incremental routing with safe reuse of the previous complete path."""
import math


class Navigator:
    def __init__(self, geo):
        self.geo = geo
        self.path = []
        self.search = None
        self.goal = None
        self.replan_at = -math.inf

    def invalidate(self):
        self.path = []
        self.search = None
        self.replan_at = -math.inf

    def move(self, pos, goal, now, budget_ms=8, speed=1):
        changed = self.goal is None or math.dist(goal[:2],self.goal) > 10
        if changed or (self.search is None and now >= self.replan_at):
            self.search = self.geo.search(pos,goal)
            self.goal = tuple(goal[:2])
        if self.search is not None:
            status = self.search.advance(budget_ms)
            if status != 'pending':
                if status == 'complete':
                    self.path = self.search.path
                else:
                    self.path = []
                self.search = None
                self.replan_at = now+1
        while self.path and math.dist(pos[:2],self.path[0]) < .5:
            self.path.pop(0)
        # Searches can finish after the pilot moved along an older route.
        for i in range(min(len(self.path)-1,3),-1,-1):
            if self.geo.walk_clear(pos,self.path[i]):
                self.path = self.path[i:]
                break
        if not self.path or not self.geo.walk_clear(pos,self.path[0]):
            dx,dy = goal[0]-pos[0],goal[1]-pos[1]
            distance = math.hypot(dx,dy)
            scale = min(speed,distance/2)/max(distance,1e-9)
            return [dx*scale,dy*scale]
        dx,dy = self.path[0][0]-pos[0],self.path[0][1]-pos[1]
        distance = math.hypot(dx,dy)
        scale = min(speed,distance/2)/max(distance,1e-9)
        return [dx*scale,dy*scale]

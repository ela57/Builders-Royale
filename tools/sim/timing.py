"""Distinguish decision CPU work, collection pauses and wall-clock delays."""
import gc
import time


_total = [0.,0.]
_started = None
_calls = {}
_generations = [0.,0.,0.]
_slow_calls = []


def _collection(phase, info):
    global _started
    if phase == 'start':
        _started = time.perf_counter(),time.process_time()
    elif _started is not None:
        elapsed = time.perf_counter()-_started[0]
        _total[0] += elapsed
        _generations[info['generation']] += elapsed
        _total[1] += time.process_time()-_started[1]
        _started = None


gc.callbacks.append(_collection)


def collections():
    return tuple(_total)+tuple(_generations)


def trace_calls():
    import numpy as np
    from bot import Bot
    from brain import Memory
    from geo import Geo, PathSearch
    from navigation import Navigator

    def wrap(method, name):
        def measured(*args, **kwargs):
            start = time.perf_counter()
            try:
                return method(*args,**kwargs)
            finally:
                elapsed = time.perf_counter()-start
                _calls[name] = _calls.get(name,0)+elapsed
                if elapsed >= .004 and len(_slow_calls) < 20:
                    record = {'name':name,'ms':round(elapsed*1000,3)}
                    if name == 'Geo.los':
                        record['ray'] = [[float(v) for v in point] for point in args[1:3]]
                    elif name.startswith('numpy.'):
                        record['shapes'] = [list(value.shape) for value in args if hasattr(value,'shape')]
                    _slow_calls.append(record)
        return measured

    for cls,names in ((Bot,('_cover','_zone_goal','_explore','_loot_goal','_avoid_npcs')),
                      (Memory,('update',)),(Geo,('ground','nearest','walk_clear','los','cover_score','find_cover')),
                      (Navigator,('move','invalidate')),(PathSearch,('advance','close'))):
        for name in names:
            setattr(cls,name,wrap(getattr(cls,name),cls.__name__+'.'+name))
    for owner,names in ((np,('asarray','linspace','floor','any','all','max','clip','minimum')),(np.linalg,('norm',))):
        for name in names:
            setattr(owner,name,wrap(getattr(owner,name),'numpy.'+name))


def reset_calls():
    _calls.clear()
    _slow_calls.clear()


def calls():
    return {name:round(value*1000,3) for name,value in _calls.items()} | {'slow_calls':_slow_calls[:]}

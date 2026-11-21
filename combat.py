"""Weapon geometry, charge sequencing and defensive movement."""
import math
import time


def angle_delta(a, b):
    return (a-b+math.pi) % (2*math.pi)-math.pi


def unit(dx, dy):
    length = math.hypot(dx,dy)
    return [dx/max(length,1e-9),dy/max(length,1e-9)]


def intercept(origin, target, velocity, speed):
    """Constant-velocity intercept in 3D; None when no forward solution exists."""
    r = [target[i]-origin[i] for i in range(3)]
    a = sum(v*v for v in velocity)-speed*speed
    b = 2*sum(r[i]*velocity[i] for i in range(3))
    c = sum(v*v for v in r)
    if speed <= 0:
        return None
    if abs(a) < 1e-9:
        roots = [-c/b] if abs(b) > 1e-9 else ([0] if c == 0 else [])
    else:
        disc = b*b-4*a*c
        roots = [] if disc < 0 else [(-b-math.sqrt(disc))/(2*a),(-b+math.sqrt(disc))/(2*a)]
    forward = [t for t in roots if t >= 0]
    if not forward:
        return None
    t = min(forward)
    return tuple(target[i]+velocity[i]*t for i in range(3))


def centre(actor):
    pos = actor['pos']
    height = 1.75 if actor.get('kind') == 'npc' else 4.5*(1+.05*actor.get('cubes',0))
    if actor.get('kind') == 'crate':
        height = 1.2
    return (pos[0],pos[1],pos[2]+height)


def aim_at(origin, target):
    dx,dy,dz = (target[i]-origin[i] for i in range(3))
    return [math.atan2(dy,dx),math.atan2(dz,math.hypot(dx,dy))]


def aim_error(actual, desired):
    return math.hypot(angle_delta(actual[0],desired[0]),actual[1]-desired[1])


def best_weapon(me, distance):
    weapons = [w for w in me.get('weapons',[]) if w]
    lightning = next((w for w in weapons if w['kind'] == 'lightning' and distance < .9*w['stats'].get('range',45)),None)
    return lightning or next((w for w in weapons if w['kind'] == 'laser'),None) or next((w for w in weapons if w['kind'] == 'gun'),None)


def fire_control(me, target, visible, actors, geo, config, deadline=math.inf):
    """Return desired torso aim, slot triggers, reload; obs charge is authoritative."""
    weapons = me.get('weapons',[])
    fire = [False,False]
    reload = any(w and w['kind'] == 'gun' and (w.get('ammo',0) == 0 or (w.get('ammo',12) < 4 and target is None)) for w in weapons)
    if target is None:
        return None,fire,reload
    pos = me['pos']
    muzzle = (pos[0],pos[1],pos[2]+me.get('stats',{}).get('muzzle_height',7))
    point = centre(target)
    distance = math.dist(muzzle,point)
    selected = best_weapon(me,distance)
    desired = point
    velocity = target.get('vel',[0,0,0])
    if selected and selected['kind'] == 'gun':
        desired = intercept(muzzle,point,velocity,selected['stats'].get('speed',252)) or point
    aim = aim_at(muzzle,desired)
    if not visible or not geo.los(muzzle,point):
        return aim,fire,reload
    actual = me.get('aim',[0,0])
    energy = me.get('energy',0)
    for slot,w in enumerate(weapons[:2]):
        if time.perf_counter() >= deadline:
            break
        if not w or w.get('cooldown',0) > 0:
            continue
        stats,kind = w['stats'],w['kind']
        if distance >= stats.get('range',0):
            continue
        if kind == 'gun':
            lead = intercept(muzzle,point,velocity,stats.get('speed',252))
            if lead is not None and geo.los(muzzle,lead):
                fire[slot] = w.get('ammo',0) > 0 and not w.get('reload',0) and aim_error(actual,aim_at(muzzle,lead)) < math.atan(2.5/max(distance,1))+.02
        elif kind == 'laser':
            charge = w.get('charge',0)
            aligned = aim_error(actual,aim_at(muzzle,point)) < math.atan(2/max(distance,1))
            fire[slot] = charge < stats.get('charge',.75) or (not aligned and charge < stats.get('charge',.75)+stats.get('hold_s',2)-.1)
        elif kind == 'lightning':
            cone = math.radians(stats.get('cone_deg',30))
            in_cone = abs(angle_delta(actual[0],aim_at(muzzle,point)[0])) <= cone
            nearest = True
            if in_cone:
                for other in actors:
                    if time.perf_counter() >= deadline:
                        nearest = False
                        break
                    if other['id'] == target['id']:
                        continue
                    p = centre(other)
                    if math.dist(muzzle,p) < distance and abs(angle_delta(actual[0],aim_at(muzzle,p)[0])) <= cone and geo.los(muzzle,p):
                        nearest = False
                        break
            cost = stats.get('energy',3)
            fire[slot] = distance < .9*stats.get('range',45) and in_cone and nearest and energy >= cost
            if fire[slot]:
                energy -= cost
    return aim,fire,reload


def dodge(me, enemies, projectiles, geo, config, deadline=math.inf):
    pos = me['pos']
    for enemy in enemies:
        if time.perf_counter() >= deadline:
            return None
        bearing = math.atan2(pos[1]-enemy['pos'][1],pos[0]-enemy['pos'][0])
        if enemy.get('charging') and abs(angle_delta(enemy.get('aim',[0,0])[0],bearing)) < .15:
            return unit(-(pos[1]-enemy['pos'][1]),pos[0]-enemy['pos'][0])
    gravity = config.get('weapons',{}).get('mortar',{}).get('gravity',22)
    for shell in projectiles:
        if time.perf_counter() >= deadline:
            break
        if shell.get('kind') != 'mortar':
            continue
        p,v = shell['pos'],shell['vel']
        ground = float(geo.ground(p[0],p[1]))
        for _ in range(2):
            disc = v[2]*v[2]+2*gravity*max(0,p[2]-ground)
            t = (v[2]+math.sqrt(disc))/max(gravity,1e-6)
            landing = [p[0]+v[0]*t,p[1]+v[1]*t]
            ground = float(geo.ground(*landing))
        if math.dist(pos[:2],landing) < 12:
            away = unit(pos[0]-landing[0],pos[1]-landing[1])
            return away if any(away) else [1,0]
    return None

import math, random
import config as cfg


def vanilla_pso(fobj, dim, lo=0.0, hi=None, n=None, iters=None):
    """Standard PSO. Curve is monotone (only records improvements)."""
    hi    = hi if hi is not None else float(cfg.MC)
    n     = n     or cfg.N_PARTICLES
    iters = iters or cfg.MAX_ITER
    vmax  = 0.3 * (hi - lo)

    pos   = [[random.uniform(lo, hi) for _ in range(dim)] for _ in range(n)]
    vel   = [[random.uniform(-vmax, vmax) for _ in range(dim)] for _ in range(n)]
    pbest = [p[:] for p in pos]
    pfit  = [fobj(p) for p in pos]
    gi    = min(range(n), key=lambda i: pfit[i])
    gbest, gbest_f = pbest[gi][:], pfit[gi]
    curve = []

    for _ in range(iters):
        for i in range(n):
            for d in range(dim):
                r1, r2 = random.random(), random.random()
                vel[i][d] = (0.729 * vel[i][d]
                             + 1.494 * r1 * (pbest[i][d] - pos[i][d])
                             + 1.494 * r2 * (gbest[d]    - pos[i][d]))
                vel[i][d] = max(-vmax, min(vmax, vel[i][d]))
                pos[i][d] = max(lo,   min(hi,   pos[i][d] + vel[i][d]))
            f = fobj(pos[i])
            if f < pfit[i]:
                pbest[i], pfit[i] = pos[i][:], f
            if f < gbest_f:
                gbest, gbest_f = pos[i][:], f
        curve.append(-gbest_f)   

    return gbest, gbest_f, curve


def asa_pso(fobj, dim, lo=0.0, hi=None, n=None, iters=None):
    hi    = hi if hi is not None else float(cfg.MC)
    n     = n     or cfg.N_PARTICLES
    iters = iters or cfg.MAX_ITER
    vmax  = 0.3 * (hi - lo)

    pos   = [[random.uniform(lo, hi) for _ in range(dim)] for _ in range(n)]
    vel   = [[random.uniform(-vmax, vmax) for _ in range(dim)] for _ in range(n)]
    pbest = [p[:] for p in pos]
    pfit  = [fobj(p) for p in pos]
    gi    = min(range(n), key=lambda i: pfit[i])
    gbest, gbest_f = pbest[gi][:], pfit[gi]
    best_ever_f = gbest_f      
    best_ever   = gbest[:]
    curve = []

    T = iters / math.log(5)    

    for k in range(1, iters + 1):
        # Eq.(40) adaptive omega
        mean_f  = sum(pfit) / n
        ratio   = gbest_f / mean_f if abs(mean_f) > 1e-12 else 1.0
        omega_k = max(0.4, min(0.95, 0.9 - (k / iters) * (1.0 - ratio)))

        
        c1 = 2.5  - k * (2.5  - 1.0) / iters
        c2 = 1.0 + k * (2.5  - 1.0) / iters

        for i in range(n):
            j     = random.choice([x for x in range(n) if x != i])
            cbest = [random.random() * (pbest[i][d] - pbest[j][d])
                     for d in range(dim)]

            p_cb   = math.exp(-abs(pfit[i] - pfit[j]) / max(T, 1e-9))
            use_cb = random.random() < p_cb

            for d in range(dim):
                r1, r2 = random.random(), random.random()
                social = cbest[d] if use_cb else (gbest[d] - pos[i][d])
                vel[i][d] = (omega_k * vel[i][d]
                             + c1 * r1 * (pbest[i][d] - pos[i][d])
                             + c2 * r2 * social)
                vel[i][d] = max(-vmax, min(vmax, vel[i][d]))
                pos[i][d] = max(lo,   min(hi,   pos[i][d] + vel[i][d]))

            f = fobj(pos[i])
            if f < pfit[i]:
                pbest[i], pfit[i] = pos[i][:], f

            if f <= gbest_f:
                gbest, gbest_f = pos[i][:], f
            elif T > 1e-6 and random.random() < math.exp(-(f - gbest_f) / T):
                gbest, gbest_f = pos[i][:], f   

            if f < best_ever_f:
                best_ever_f = f
                best_ever   = pos[i][:]

        T *= cfg.SA_MU                   
        curve.append(-best_ever_f)           

    return best_ever, best_ever_f, curve
import cvxpy as cp
import numpy as np

def MEKC(graph, limit = 10, solver=cp.SCIP, solver_options=None, **kwargs):
    if not solver:
        solver = cp.GLPK_MI
    if not solver_options:
        solver_options = {}
    
    completed = set()
    edges = []
    for i,k in graph.items():
        for j,v in k.items():
            if j not in completed:
                edges.append((i,j,v))
        completed.add(i)
        
    r_index = {k:i for i,k in enumerate(list(graph.keys()))}
    
    scores = [e[-1] for e in edges]
    x_vars = cp.Variable(len(graph), boolean=True)
    e_vars = cp.Variable(len(edges), boolean=True)
    constraints = [cp.sum(x_vars) == limit, cp.sum(e_vars) == limit*(limit-1)/2]
    constraints += [e_vars[i] <= x_vars[r_index[edges[i][0]]] for i in range(len(edges))]
    constraints += [e_vars[i] <= x_vars[r_index[edges[i][1]]] for i in range(len(edges))]
    objective = cp.Maximize(cp.sum(scores @ e_vars))
    
    
    try:
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver, verbose=False, **solver_options)
        index = np.array(np.round(x_vars.value), dtype=bool)
        check = np.sum(index)
        if check != limit:
            print("Error in obtaining k-clique")
            print(x_vars.value)
            return [], []
        else:
            return index, \
                np.sum(np.array(scores)*np.array(e_vars.value, dtype=bool))/(limit*(limit-1)/2)
    except:
        print("Error in solving")
        return [], []
    
def star_heuristic(g, threshold, size_limits):
    theta_l, theta_u = threshold
    kappa_l, kappa_u = size_limits
    g = {i:{j:v for j,v in k.items() if v >= theta_l and v<=theta_u and j in g} for i,k in g.items()}
    to_keep = {i for i in g if len(g[i]) >= kappa_l}
    g = {i:{j:v for j,v in g[i].items() if j in to_keep} for i in to_keep}
    
    stars = sorted([(i, g[i].keys(),np.mean(list(g[i].values()))
                     if len(g[i])>0 else -1)
                    for i in g],
                key=lambda x:x[-1], reverse=True)
    groups = []
    selected = set()
    for center, neighbours, score in stars:
        if center in selected: continue
        indices = [x for x in neighbours if x not in selected]
        
        if len(indices) >= kappa_u:
            shortlist = sorted([(i,g[center][i]) for i in indices],
                             key=lambda x:x[-1], reverse=True)
            indices = [shortlist[i][0] for i in range(kappa_u-1)]
        indices.append(center)
        
        if len(indices) >= kappa_l:
            groups.append(indices)
        selected.update(indices)
    return groups

def MMEKC(g, thresholds=(0,1), size_limits=(10,32), time_limit_s=180):
    isets = star_heuristic(g, thresholds, size_limits)
    results = []
    g_primes = []
    for g_prime in isets:
        remaining = set(g_prime)
        g_primes.append({i:{j:g[i][j] for j in remaining if j != i} for i in remaining})
    del g
    for g_prime in g_primes:
        chosen, score = MEKC(g_prime, solver=cp.SCIP, solver_options={"limits/time": time_limit_s})
        topics = np.array(list(g_prime.keys()))[chosen]
        results.append((topics, score))
        
    return results, isets
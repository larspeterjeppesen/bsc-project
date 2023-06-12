import numpy as np
rng = np.random.default_rng()

def GetEdges(A):
  edges = []
  for i in range(A.shape[1]):
    for j in range(A.shape[0]):
      if A[j,i]:
        edges.append((i,j))
  return edges


def UsePathAlg1(P,p):
  n = len(P)
  P_samples = np.zeros((n,n))
  Ns = np.zeros((n,n))
  path_cost = 0
  for i in range(len(p)-1):
    a = p[i]
    b = p[i+1]
    x = 1 if rng.random() < P[a]*P[b] else 0
    P_samples[b,a] += x
    path_cost += x
    Ns[b,a] += 1 
  return (P_samples, Ns, path_cost)
  

def UsePathCombLCB(P,p):
  n = len(p)
  Ps = np.zeros(len(P))
  Ns = np.zeros(len(P))
  c = 0
  for i in range(n-1):
    n1 = p[i]
    n2 = p[i+1] 
    P_a = 1 if rng.random() < P[n1] else 0
    P_b = 1 if rng.random() < P[n2] else 0
    Ps[n1] += P_a
    Ns[n1] += 1
    Ps[n2] += P_b
    Ns[n2] += 1
    c += P_a*P_b
  return (Ps, Ns, c)

def UsePathTS(P,p):
  n = len(p)
  Ps = np.zeros((len(P),2))
  Ns = np.zeros(len(P))
  c = 0
  for i in range(n-1):
    n1 = p[i]
    n2 = p[i+1]
    P_a = 1 if rng.random() < P[n1] else 0
    P_b = 1 if rng.random() < P[n2] else 0
    Ns[n1] += 1
    Ns[n2] += 1
    a = [1,0] if P_a else [0,1]
    b = [1,0] if P_b else [0,1]
    Ps[n1] += np.asarray(a)
    Ps[n2] += np.asarray(b)
    c += P_a*P_b
  return (Ps, Ns, c)

def UsePathESCB(P,p):
  n2 = len(p)
  n = int(n2**0.5)
  Ps = np.zeros(len(P))
  Ns = np.zeros(len(P))
  c = 0
  for i in range(n2):
    if p[i]:
      b = int(i/n)
      a = i%n
      P_a = 1 if rng.random() < P[a] else 0
      P_b = 1 if rng.random() < P[b] else 0
      Ps[a] += P_a
      Ps[b] += P_b
      Ns[a] += 1
      Ns[b] += 1
      c += P_a*P_b
  return (Ps, Ns, c)


def GetBestPath(E,P,dest):
  E = [(a,b,P[a]*P[b]) for (a,b) in E]
  n = len(P)
  dist = np.zeros(n) + float('inf')
  dist[0] = 0
  p = np.zeros(n, dtype=int) - 1

  while(True):
    modified = False
    for (a,b,cost) in E:
      if dist[a] < float('inf'):
        if dist[b] > dist[a] + cost:
          dist[b] = dist[a] + cost
          p[b] = a
          modified = True
    if modified == False:
      break

  best_path = [n-1]
  while(True):
    i = p[best_path[0]]
    best_path.insert(0,i)
    if i == 0:
      break

  return (best_path,dist[dest])

def PathToList(path):
  n = int(len(path)**0.5)
  path = path.reshape((n,n))
  p = []
  a = 0
  b = 0
  while(True):
    if path[b,a]:
      p.append(a)
      if b == n-1:
        p.append(b)
        break
      a = b
      b = 0
    else:
      b += 1
  return p


def AsConPath(path, n):
  conpath = np.zeros(n*n) 
  for i in range(len(path)-1):
    a = path[i] #From node, column in weight Matrix
    b = path[i+1] #To node, row in weight Matrix
    index = b*n + a
    conpath[index] = 1
  return conpath

def PathListAsMatrix(paths,n):
  return np.asarray([AsConPath(path,n) for path,c in paths])

#Returns a list tuples ([path], cost)
def ListPaths(A, P, source, dest):
  n = A.shape[0]
  P = P.reshape((len(P),1))
  C = P@P.T

  paths = [[([source],0)]]
  #This double loop creates a list of non-cyclic 
  #paths originating from source to all destinations
  for i in range(n):
    paths.append([])
    for path in paths[i]:
      endpoint = path[0][len(path[0])-1]
      neighbors = A[:,endpoint] #nodes that endpoint connects to
      neighborCosts = C[:,endpoint] #weights of neighbors
      for index in range(n):
        if (neighbors[index] and index not in path[0]):
          c = neighborCosts[index]
          paths[i+1].append((path[0] + [index], path[1]+c))
  #flatten list
  paths = [path for sublist in paths for path in sublist]
  #sort out paths that don't end at the destination 
  paths = list(filter(lambda p: p[0][len(p[0])-1] == dest, paths))
  #sort paths by cost
  paths = sorted(paths, key=lambda tup: tup[1], reverse=True)
  return paths


def RankPath(Path_list, path):
  n = len(Path_list)
  for i in range(n-1, -1, -1):
    if Path_list[i][0] == path:
      return i
  return None


def EdgeToNodeUsage(Ns):
    Usage = np.zeros(Ns.shape[0])
    
    for i in range(Ns.shape[0]):
        for j in range(Ns.shape[1]):
            Usage[i] += Ns[i,j]
            Usage[j] += Ns[i,j]
            
    return Usage

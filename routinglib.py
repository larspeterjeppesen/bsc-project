import numpy as np
rng = np.random.default_rng()

def GetEdges(A):
  edges = []
  for i in range(A.shape[1]):
    for j in range(A.shape[0]):
      if A[j,i]:
        edges.append((i,j))
  return edges

def BellmanFordPath(E, Cs, Ns, t, CostFunc):
  E = [(a,b,CostFunc(Cs, Ns, a, b, t)) for (a,b) in E]
  n = Cs.shape[0]
  dist = np.zeros(n) + float('inf')
  dist[0] = 0
  p = np.zeros(n, dtype=int) - 1
  
  for i in range(n-1):
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
    best_path.insert(0, i)
    if i == 0:
      break     
  return best_path

def GetBestPath(E,C,dest):
  E = [(a,b,C[b,a]) for (a,b) in E]
  n = C.shape[0]
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

def UsePathLCB(C,p):
  n = C.shape[0]
  C_samples = np.zeros((n,n))
  Ns = np.zeros((n,n))
  path_cost = 0
  for i in range(len(p)-1):
    a = p[i]
    b = p[i+1]
    x = 1 if rng.random() < C[b,a] else 0
    C_samples[b,a] += x
    path_cost += x
    Ns[b,a] += 1 
  return (C_samples, Ns, path_cost)

def UsePathTS(C,p):
  n = len(p)
  (x,y) = C.shape
  Cs = np.zeros((x,y,2))
  c = 0
  for i in range(n-1):
    a = p[i]
    b = p[i+1]
    x = 1 if rng.random() < C[b,a] else 0
    if x:
      Cs[b,a,0] += 1
    else:
      Cs[b,a,1] += 1
    c += x
  return (Cs, c)

def UsePathESCB(C,p):
  n2 = len(p)
  n = int(n2**0.5)
  Cs = np.zeros((n,n))
  Ns = np.zeros((n,n))
  c = 0
  for i in range(n2):
    if p[i]:
      b = int(i/n)
      a = i%n
      x = 1 if rng.random() < C[b,a] else 0
      Cs[b,a] += x 
      Ns[b,a] += 1
      c += x
  return (Cs, Ns, c)


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

#Breadth first search
#Returns a list tuples ([path], cost)
def ListPaths(A, C):
  n = A.shape[0]
  source = 0
  dest = n-1

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

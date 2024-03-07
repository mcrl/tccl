import copy
def f(pair):
  print(id(pair))
  pair = copy.deepcopy(pair)
  print(id(pair))
  def helper():
    print(id(pair))
    pair.append(1)
  helper()
  return pair

print(f([1,2,3]))
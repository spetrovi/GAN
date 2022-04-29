import json

with open('e1.json', 'r') as fileobject:
    x = json.load(fileobject)
    
def my_sort(t):
    return t[0]
    
a = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
for k,v in a.items():
    print(k,v)

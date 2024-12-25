names = ['test', 'tist', 'tost']

objects_co = {}
for name in names:
    objects_co[name] = {'name':name, 'co':[]}
    
objects_co['test']['co'].append('test')

print(objects_co)
for key in objects_co:
    print(objects_co[key])
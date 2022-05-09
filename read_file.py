import numpy as np
path="winner-feedforward.gv"
with open(path) as file:
	contents = file.read()
	#print(type(contents))
	lines=contents.split('\n')
	# print(type(lines))
	tiles=[]
	for l in lines:
		if l.find("->")!=-1:
			if l.find('t') == 1:
				tiles.append(l[1:6])
	# print(tiles)
	file.close()
	print(len(tiles))
	tiles=list(set(tiles))
	print(len(tiles))
np.save('selected-features.npy',np.array(tiles))
import os

from path import Path
root= Path("/wrk/users/nicjosw/nyu_untouched")
images=[folder[:-1] for folder in open(Path("nyu_eval/nyu_val_split.txt"))]
triplets=[]
for image in images:
    scene=image.split("/")[0]
    img=image.split("/")[1]
    print(scene)
    print(str(root/scene))

    img_data = sorted([fn for fn in os.listdir(root/scene)
              if "r-" in fn])
    #print(img_data)
    scene_length = len(img_data)
    index=img_data.index(img)
    if index-10<0:
        image_indize=[index+10,index,index+10]
    elif index+10>=scene_length:
        image_indize=[index-10,index,index-10]
    else:
        image_indize=[index-10,index,index+10]
    
    a,b,c=img_data[image_indize[0]],img_data[image_indize[1]],img_data[image_indize[2]]
    image_paths=[a,b,c]
    image_paths=[scene+"/"+path for path in image_paths]
    triplets.append(image_paths[0])
    triplets.append(image_paths[1])
    triplets.append(image_paths[2])

with open('nyu_eval/sl3eval.txt', 'w') as f:
    for item in triplets:
        f.write("%s\n" % item)

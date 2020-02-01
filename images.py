
import pickle
import vocabulary
from PCV.localdescriptors import sift


imlist = [
    # 'images/1.JPG',
    # 'images/2.JPG',
    # 'images/3.JPG',
    # 'images/4.JPG',
    # 'images/5.JPG',
    # 'images/6.JPG',
    # 'images/7.JPG',
    # 'images/8.JPG',
    'images/9.JPG',
    'images/10.JPG',
]

nbr_images = len(imlist)


featlist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]

for i in range(nbr_images):
    # print featlist[i]
    # print imlist[i]
    sift.process_image(imlist[i],featlist[i])

"""

imagename = "/opt/cv/images/2603.JPG"
from PIL import Image
im = Image.open(imagename).convert('L')
im.save('/opt/cv/images/tmp.pgm')
sift /opt/cv/images/tmp.pgm  --output /opt/cv/images/2603.sift --edge-thresh 10 --peak-thresh 5
"""
# print "ok"
# exit() 

voc = vocabulary.Vocabulary('ukbenchtest') 
voc.train(featlist, 1000, 10)

with open('vocabulary.pkl', 'wb') as f:
    pickle.dump(voc, f)
    
print('vocabulary is:', voc.name, voc.nbr_words)


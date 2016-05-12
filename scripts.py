# http://cs.stanford.edu/people/karpathy/deepimagesent/
# make references
import json
import numpy
cap_file = json.load(open('../../Data/coco/dataset.json', 'rb'))['images']
cap_ = [[' '.join(c['tokens']) for c in cf['sentences']] for cf in cap_file]
name = ['train', 'dev', 'test']
num = [110000, 5000, 5000]
ref = cap_[num[0] + num[1]:num[0] + num[1] + num[2]]
for i,r in enumerate(ref):
    if not len(r) == 5:
        ref[i] = ref[i][:5]
ref = numpy.array(ref).transpose()
for i,r in enumerate(ref):
    print >>open('reference' + str(i), 'w'), '\n'.join(r)

# generate samples
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python generate_caps.py
# calculate score
python metrics.py gensaps.test.txt reference0 reference1 reference2 reference3 reference4
# BLEU: 0.6887/0.5034/0.3588/0.2547
# METEOR: 0.2234





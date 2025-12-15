import glob
import os
from shutil import copyfile

cinic_directory = "..\..\data\CINIC"
enlarge_directory = "..\..\data\CINIC-L"
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
sets = ['train', 'valid', 'test']
if not os.path.exists(enlarge_directory):
    os.makedirs(enlarge_directory)
if not os.path.exists(enlarge_directory + r'\train'):
    os.makedirs(enlarge_directory + r'\train')
if not os.path.exists(enlarge_directory + r'\test'):
    os.makedirs(enlarge_directory + r'\test')

for c in classes:
    if not os.path.exists(r'{}\train\{}'.format(enlarge_directory, c)):
        os.makedirs(r'{}\train\{}'.format(enlarge_directory, c))
    if not os.path.exists(r'{}\test\{}'.format(enlarge_directory, c)):
        os.makedirs(r'{}\test\{}'.format(enlarge_directory, c))

for s in sets:
    for c in classes:
        source_directory = '{}\{}\{}'.format(cinic_directory, s, c)
        filenames = glob.glob(r'{}\*.png'.format(source_directory))
        for fn in filenames:
            dest_fn = fn.split('\\')[-1]
            if s == 'train' or s == 'valid':
                dest_fn = r'{}\train\{}\{}'.format(enlarge_directory, c, dest_fn)
                copyfile(fn, dest_fn)
            elif s == 'test':
                dest_fn = r'{}\test\{}\{}'.format(enlarge_directory, c, dest_fn)
                copyfile(fn, dest_fn)
        print("----{} done.".format(c))
    print("--{} done".format(s))
print("[All done]")

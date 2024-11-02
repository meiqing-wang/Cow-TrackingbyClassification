import os
import argparse

"""
Prerequisites: input files must have the following structure
FrameID <int> ObjID <int> tlbr <4x int>

Call
python adapt_objid_gt_tr.py 
    --gt eval/tr_eval_inference.txt 
    --tr eval/tr_eval_inference.txt 
    --output_gt eval/gt_woi.txt 
    --output_tr eval/tr_woi.txt
"""

#os.chdir('/home/dalco/boris/dev_test/custom_model')

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gt', type=str, help='txt file')
    parser.add_argument('--tr', type=str, help='txt file', required=False)
    parser.add_argument('--output_gt', type=str, help='txt file')
    parser.add_argument('--output_tr', type=str, help='txt file', required=False)
    return parser.parse_args()

args = parse_args()
gt = args.gt
tr = args.tr
output_gt = args.output_gt
output_tr = args.output_tr

f_output_gt = open(output_gt, 'w')
f_output_tr = open(output_tr, 'w')

gt_cow_ids = [0,1,3,4,5,6,7,8,9,10,11,12,13]
eval_cow_ids = [] # new IDs for the evaluation

with open(gt, 'r') as f_gt, open(tr, 'r') as f_tr:
    gt_objIDs = [int(float(line.split(',')[1])) for line in f_gt]
    tr_objIDs = [int(float(line.split(',')[1])) for line in f_tr]

    gt_objIDs = list(set(gt_objIDs))
    tr_objIDs = list(set(tr_objIDs))

    union_list = list(set(gt_objIDs + tr_objIDs))
    union_list.sort() # union of the ObjIDs found in gt and tr

    for i in range(len(union_list)):
        eval_cow_ids.append(i)

    mapping_ids = dict(zip(union_list, eval_cow_ids))

    f_gt.seek(0) # go to beginning of file
    f_tr.seek(0)

    for line in f_gt:
        line = line.split(',')
        f_output_gt.write(
            ','.join([line[0],
                      str(mapping_ids.get(int(float(line[1])))),
                      str(int(float(line[2]))),
                      str(int(float(line[3]))),
                      str(int(float(line[4]))),
                      str(int(float(line[5]))) + '\n'
                      ])
            )

    for line in f_tr:
        line = line.split(',')
        f_output_tr.write(
            ','.join([line[0],
                      str(mapping_ids.get(int(float(line[1])))),
                      str(int(float(line[2]))),
                      str(int(float(line[3]))),
                      str(int(float(line[4]))),
                      str(int(float(line[5]))) + '\n'
                      ])
            )

    f_output_gt.close()
    f_output_tr.close()

    print('True object IDs:',union_list, len(union_list))
    print('Mapped object IDs', eval_cow_ids, len(eval_cow_ids))

import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--label_dir', type=str, help='dir of cvat output')
    parser.add_argument('--output',    type=str, help='txt file')
    return parser.parse_args()


def xywhcn_to_xywh(bbox, image_width=2560, image_height=1440):
    cx, cy, width, height = bbox

    x = (cx - width / 2) * image_width
    y = (cy - height / 2) * image_height
    width *= image_width
    height *= image_height

    return int(x), int(y), int(width), int(height)


def main(label_dir, output_file):
    f_eval = open(output_file, 'w')
    label_dir = os.path.join(os.getcwd(), label_dir)
    for label_file in os.listdir(label_dir):
        label_file_absolute = os.path.join(os.getcwd(), label_dir, label_file)
        frameID =  int(label_file_absolute.rstrip('.txt')[-6:])

        with open(label_file_absolute, 'r') as file:
            lines = file.read().splitlines()
            for line in lines:
                ln = line.split(' ')

                print(ln)
                #exit(1)
                
                objID = int(ln[0])
                x,y,w,h = xywhcn_to_xywh(bbox=(float(ln[1]), 
                                               float(ln[2]), 
                                               float(ln[3]), 
                                               float(ln[4])))
                
                f_eval.write(','.join([str(frameID), 
                                       str(objID), 
                                       str(x),
                                       str(y),
                                       str(w),
                                       str(h) + '\n']))


    f_eval.close()


args = parse_args()
label_dir = args.label_dir
output_file  = args.output


main(label_dir, output_file)

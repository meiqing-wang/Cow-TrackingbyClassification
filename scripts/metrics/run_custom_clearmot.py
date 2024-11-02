import os
import argparse
import numpy as np
import pandas as pd
import motmetrics as mm
from trackeval.trackeval.metrics.hota import HOTA

"""
    Taken from 'HOTA Implementation Challenges'
    https://medium.com/@jumabek4044/days-6-7-navigating-hota-implementation-challenges-6b827ae48366
"""

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gt', type=str, help='Groundtruths')
    parser.add_argument('--tr', type=str, help='Tracks')

    return parser.parse_args()


def prepare_data_for_hota(gt_data, pred_data):
    gt_df = gt_data
    pred_df = pred_data

    gt_ids_grouped = gt_df.reset_index()[['FrameID', 'ObjID']].groupby('FrameID')['ObjID'].apply(np.array)
    pred_ids_grouped = pred_df.reset_index()[['FrameID', 'ObjID']].groupby('FrameID')['ObjID'].apply(np.array)
    #print(gt_ids_grouped)
    #print(pred_ids_grouped)
    
    unique_frames = sorted(set(gt_df['FrameID']).union(set(pred_df['FrameID'])))
    #print(unique_frames)

    gt_ids_extended = [np.array([], dtype=int) for _ in range(len(unique_frames))]
    pred_ids_extended = [np.array([], dtype=int) for _ in range(len(unique_frames))]
    
    for idx, frame in enumerate(unique_frames):
        if frame in gt_ids_grouped.keys():
            gt_ids_extended[idx] = gt_ids_grouped[frame]
        if frame in pred_ids_grouped.keys():
            pred_ids_extended[idx] = pred_ids_grouped[frame]
            
    return gt_ids_extended, pred_ids_extended


def compute_tracking_metrics_for_a_sequence(gt_data, pred_data):
    print("data received for compute_tracking_metrics_for_a_sequence")
    #print(gt_data.head())
    #print(pred_data.head())

    gt_df = gt_data
    pred_df = pred_data

    acc = mm.MOTAccumulator(auto_id=True)

    frame_ids = sorted(gt_df['FrameID'].unique())

    for frame_id in range(len(frame_ids)):
        gt_frame = gt_df[gt_df['FrameID'] == frame_id]
        pred_frame = pred_df[pred_df['FrameID'] == frame_id]

        print(f'Prediction {frame_id}:', '\n', pred_frame, '\n', gt_frame)

        gt_boxes = gt_frame[['x', 'y', 'w', 'h']].to_numpy()
        pred_boxes = pred_frame[['x', 'y', 'w', 'h']].to_numpy()

        iou_matrix = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)

        acc.update(gt_frame['ObjID'], pred_frame['ObjID'], iou_matrix)

    mh = mm.metrics.create()

    # Compute MOT metrics
    summary = mh.compute(acc, 
                         metrics=['motp', 'mota', 'idf1',
                                  'mostly_tracked', 'mostly_lost', 
                                  'num_switches'], 
                         name='Metrics')
    motp = summary.loc['Metrics', 'motp']*100
    mota = summary.loc['Metrics', 'mota']*100
    idf1 = summary.loc['Metrics', 'idf1']*100
    mt   = summary.loc['Metrics', 'mostly_tracked']
    ml   = summary.loc['Metrics', 'mostly_lost']
    idsw = summary.loc['Metrics', 'num_switches']

    # Prepare data for HOTA computation
    gt_ids, pred_ids = prepare_data_for_hota(gt_df, pred_df)

    similarity_scores = []

    frame_ids = sorted(gt_df['FrameID'].unique())

    for frame_id in range(len(frame_ids)):
        gt_frame = gt_df[gt_df['FrameID'] == frame_id]
        pred_frame = pred_df[pred_df['FrameID'] == frame_id]

        #print('gt_frame',gt_frame)
        #print('pred_frame', pred_frame)
        #exit(1)

        gt_boxes = gt_frame[['x', 'y', 'w', 'h']].to_numpy()
        pred_boxes = pred_frame[['x', 'y', 'w', 'h']].to_numpy()

        #if (gt_frame['ObjID'].to_numpy() == np.array([0,1,3,4,5,2,6,7])).all():
        #    print('SALE PUTE')
        #    print('FrameID:', frame_id)

        if len(pred_boxes) > 0:
            iou_matrix = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1.0)
            similarity_scores.append(1 - iou_matrix)
        else:
            similarity_scores.append(np.zeros((len(gt_boxes), 0)))
    
        #print(similarity_scores)
        #exit(1)
            
    #print('GT IDS:',gt_ids)

    data = {
        'num_gt_dets': len(gt_df),
        'num_tracker_dets': len(pred_df),
        'num_gt_ids': max(gt_df['ObjID'].unique()) + 1,
        'num_tracker_ids': max(pred_df['ObjID'].unique()) + 1,
        'gt_ids': gt_ids,
        'tracker_ids': pred_ids,
        'similarity_scores': similarity_scores
    }

    # Compute HOTA
    hota_metric = HOTA()
    hota_result = hota_metric.eval_sequence(data)
    hota_score = hota_result['HOTA'][0] * 100

    return {'motp': round(motp,2), 
            'mota': round(mota,2), 
            'idf1': round(idf1,2),
            'hota': round(hota_score,2),
            'mt': mt,
            'ml': ml,
            'idsw': idsw}


# MAIN PART
args = parse_args()
gt = pd.read_csv(args.gt, sep=',', usecols=[i for i in range(6)], header=None)
tr = pd.read_csv(args.tr, sep=',', usecols=[i for i in range(6)], header=None)


gt_df = gt
pred_df = tr

# FOR DEEPSORT, PLEASE PUT THOSE TWO LINES IN COMMENTS !!!
#pred_df.iloc[:,4] = pred_df.iloc[:,4] - pred_df.iloc[:,2]
#pred_df.iloc[:,5] = pred_df.iloc[:,5] - pred_df.iloc[:,3]



gt_df.columns = ['FrameID', 'ObjID', 'x', 'y', 'w', 'h']
pred_df.columns = ['FrameID', 'ObjID', 'x', 'y', 'w', 'h']

# Here [x1,y1,x2,y2] to [x1,y1,w,h]


# PLEASE ADJUST THOSE!
#gt_df['FrameID'] = gt_df['FrameID'] - 1
#pred_df['FrameID'] = pred_df['FrameID'] - 1
#gt_df['ObjID'] = gt_df['ObjID'] - 1
#pred_df['ObjID'] = pred_df['ObjID'] - 1

for i in gt_df.columns:
    try:
        gt_df[[i]] = gt_df[[i]].astype(float).astype(int)
    except:
        pass

for i in pred_df.columns:
    try:
        pred_df[[i]] = pred_df[[i]].astype(float).astype(int)
    except:
        pass


res = compute_tracking_metrics_for_a_sequence(gt_data=gt_df, pred_data=pred_df)
print(res)
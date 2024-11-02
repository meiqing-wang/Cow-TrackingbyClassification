"""
    This is the script to call when you want to run a real-time inference
    on a video of cows.

    Call:
        python inference.py \
            -i vid95_1_20230824_141834_cut.mp4 \
            -o test.mp4

"""
import os
import cv2
import argparse
import numpy as np
import seaborn as sns
from ultralytics import YOLO
from time import localtime, strftime # time
from scipy.optimize import linear_sum_assignment as linear_assignment


def crop_roi(frame, bbox):
    """
    Returns the RoI specified by the bounding box.
    """
    x_min, y_min, x_max, y_max = bbox

    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_max)
    y_max = int(y_max)

    #print('This is frame', frame.shape)
    #print('This is bbox:', bbox)

    roi = frame[y_min:y_max, x_min:x_max]

    #cv2.imwrite('test.png', roi) 
    #print('This is roi:',roi.shape)

    return roi


def tlbr_to_tlwh(boxes):
    """
    Converts [x1,y1,x2,y2] to [x,y,w,h].
    """
    converted_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        converted_boxes.append([x1, y1, w, h])
    return converted_boxes


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(x1, y1, x2, y2)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher prob means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    # get x1,y1 and x2,y2
    bbox_tl, bbox_br = bbox[:2], bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, 2:]

    #print(candidates)

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    # bbox
    wb = bbox[2] - bbox[0]
    hb = bbox[3] - bbox[1]

    # candidates
    x1, y1, x2, y2 = np.split(candidates, 4, axis=1)
    w = x2 - x1
    h = y2 - y1
    transformed_array = np.concatenate((x1, y2, w, h), axis=1)
    #print(transformed_array)

    area_intersection = wh.prod(axis=1)
    area_bbox = wb*hb
    area_candidates = transformed_array[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def main(video_input, video_output, output_file, threshold_1, threshold_2):
    """
    Reads a video frame-by-frame and does the tracking.
    Store the video output.
    Save txt file.
    """
    # rad first frame
    cap = cv2.VideoCapture(video_input)
    ret, frame = cap.read()

    # load models
    model_detector = YOLO('../weights/detection/yolov8x_best.pt')
    model_classifier = YOLO('../weights/classification_2/yolov8l-cls_best.pt')
    print('Classifier weights loaded.')
    ids = model_classifier.names
    #print('ids:', ids)
    #inv_ids = {v: k for k, v in ids.items()}

    # create output video
    cap_out = cv2.VideoWriter(video_output, 
                              cv2.VideoWriter_fourcc(*'mp4v'), 
                              cap.get(cv2.CAP_PROP_FPS), # get the FPS
                              (frame.shape[1], frame.shape[0])) 

    # set colors
    palette = sns.color_palette(None, 200)
    palette = [(int(pal[0]*255), int(pal[1]*255), int(pal[2]*255)) 
                for pal in palette]

    # set frame counter
    counter = 0

    # save txt files
    f_det = open('eval/mat/detections.csv', 'w')
    f_mat_lap_first = open('eval/mat/lap_first_matrix.csv', 'w')

    f_mat_matched = open('eval/mat/lap_matrix_matched.csv', 'w')
    f_mat_unmatched = open('eval/mat/lap_matrix_unmatched.csv', 'w')

    f_output = open(output_file, 'w')


    while ret:
        print(f'[INFO] Frame {counter}', '-', strftime("%H:%M:%S", localtime()))

        # bbox predictions
        results = model_detector.predict(frame, conf = 0.6, verbose=False)

        detections = []
        conf_probs = []
        roi_list = []
        pred_ids_list = []
        
        for result in results:
            for r in result.boxes.conf:
                conf_probs.append(round(r.cpu().tolist(), 2))
            for r in result.boxes.xyxy.tolist():
                x1,y1,x2,y2 = r
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                detections.append([x1,y1,x2,y2])

            #print('detections', detections, type(detections), sep = '\n')

            for det in detections:
                roi_list.append(crop_roi(frame, det))

        class_prob_for_detections = []

        # class predictions
        for i in range(len(roi_list)):
            results = model_classifier.predict(roi_list[i], verbose = False)
            for result in results:
                pred_ids_list.append(ids[result.probs.top1])
                class_prob_for_detections.append(np.asarray(result.probs.cpu().numpy().data))

        #print('Detections VS. Classification probabilities:', 
        #      np.round(np.asarray(class_prob_for_detections), 2),
        #      sep = '\n') 
        #print('')


        # 1st lap matrix
        class_prob_for_detections_array = np.array(class_prob_for_detections)
        
        if class_prob_for_detections_array.size == 0:
            # write video
            cap_out.write(frame)
            # next frame
            ret, frame = cap.read()
            counter += 1	
            continue

        # 1st classifcation threshold
        mask_1 = np.any(class_prob_for_detections_array >= threshold_1, axis = 1)
        #print(mask_1)

        matched_matrix = class_prob_for_detections_array[mask_1]
        unmatched_matrix = class_prob_for_detections_array[~mask_1]
        #print(  'matched:',   matched_matrix.round(2), sep = '\n')
        #print('unmatched:', unmatched_matrix.round(2), sep = '\n')
        matched_detections = np.array(detections)[mask_1]
        unmatched_detections = np.array(detections)[~mask_1]


        # 1st linear assigment
        row_id, col_id = linear_assignment(matched_matrix, maximize = True) # maximization


        # retrieve classification probabilities
        classification_prob = matched_matrix[row_id, col_id]
        classification_prob = classification_prob.astype(float)
        #print('classification_prob:', classification_prob)
        #print('')


        # retrieve attributed IDs
        matched_ids_string = [ids.get(key) for key in col_id]
        matched_ids = [int(id.split('_')[1]) for id in matched_ids_string]


        #print('matched_ids_string:', matched_ids_string)
        #print('matched_ids:', matched_ids)
        #print('col_id:', col_id)


        # evalutation
        frames_nr = np.array([[counter] * len(detections)])
        frames_nr_matched = np.array([[counter] * len(matched_detections)])
        conf_probs = np.array(conf_probs)
        matched_detections_tlwh = np.array(tlbr_to_tlwh(matched_detections))
        predicted_ids = np.array(matched_ids)

        # detections with confidence score
        detections_with_conf_score = np.concatenate((frames_nr.reshape(-1, 1),
                                                    np.array(detections),
                                                    conf_probs.reshape(-1, 1)),
                                                    axis=1)
        np.savetxt(f_det, detections_with_conf_score, delimiter = ',', 
                   fmt = '%i,%i,%i,%i,%i,%.2f')

        # matrix first LAP
        f_mat_lap_first.write(f'Frame {counter}\n')
        np.savetxt(f_mat_lap_first, np.array(class_prob_for_detections), delimiter = ' ', fmt = '%.2f')
        f_mat_lap_first.write('\n')
        f_mat_lap_first.write('\n')

        # matrix matches after first LAP
        f_mat_matched.write(f'Frame {counter}\n')
        np.savetxt(f_mat_matched, np.array(matched_matrix), delimiter = ' ', fmt = '%.2f')
        f_mat_matched.write('\n')
        f_mat_matched.write('\n')

        # matrix unmatches after first LAP
        f_mat_unmatched.write(f'Frame {counter}\n')
        np.savetxt(f_mat_unmatched, np.array(unmatched_matrix), delimiter = ' ', fmt = '%.4f')
        f_mat_unmatched.write('\n')
        f_mat_unmatched.write('\n')


        # 2nd assignment
        """
        ids := dictionary with key 0 to 12, and corresponding 'cow_0' to 'cow_12'
        idx := column ID of the LAP matrix, matching with ids

        ids_remaining := dictionary with key 0 to N remaining animals

        adjusted_unmatched_matrix := 
            matrix with unmatched rows and unmatched remaining columns, 
            resulting from the remaining row_id and col_id from LAP

        remaining_matrix :=
            matrix where rows meet the 0.4 threshold,
            row_id in this matrix will still be matched.

        unmatched_detections :=
            matrix of detections [x1, y1, x2, y2] which correspond to 
            the rows of adjusted matrix (!)
        """
        # remove already assigned IDs from unmatched_detections
        matched_matrix_idx = np.sort(col_id)
        unmatched_matrix_idx = [idx for idx in range(matched_matrix.shape[1])
                                if idx not in matched_matrix_idx]

        ids_remaining = {_:ids.get(key) for _, key in enumerate(unmatched_matrix_idx)}


        #print('matched_matrix_idx:', matched_matrix_idx)
        #print('unmatched_matrix_idx:', unmatched_matrix_idx)
        #print('ids_remaining:', ids_remaining)


        # 1st lap matrix without attributed columns
        adjusted_unmatched_matrix = np.delete(unmatched_matrix, col_id, axis = 1)


        # if detections are still present
        if adjusted_unmatched_matrix.shape[0] > 0:
            """Consider unmatched detections."""

            # 2nd classifcation threshold
            mask_2 = np.any(adjusted_unmatched_matrix >= threshold_2, axis = 1)

            #print('ADJUSTED_UNMATCHED:', adjusted_unmatched_matrix.shape[0])
            #print('ADJUSTED_UNMATCHED:', adjusted_unmatched_matrix)

            # 2nd lap matrix
            remaining_matrix = adjusted_unmatched_matrix[mask_2]

            # remaining detections
            remaining_detections = unmatched_detections[mask_2]

            #print('REMAINING MAT:', remaining_matrix)
            #print('REMAINING DET:', remaining_detections)
            #print('SHAPE:', remaining_matrix.shape[0])

            if remaining_matrix.shape[0] > 1:
                """Consider unmatched detections where at least one classification score
                   is above the second specified threshold.
                """
                # 2nd linear assignment
                row_id_remaining, col_id_remainig = \
                    linear_assignment(remaining_matrix, maximize = True)

                # 2nd ids, prob, and dets
                remaining_ids_to_match = [ids_remaining.get(key) for key in col_id_remainig]
                remaining_ids_to_match = [int(id.split('_')[1]) for id in remaining_ids_to_match]
                
                remaining_class_prob = remaining_matrix[row_id_remaining, col_id_remainig]
                remaining_class_prob = remaining_class_prob.astype(float)

                remaining_dets_to_match = remaining_detections[row_id_remaining]
                #print('REMAINING IDS:', remaining_ids_to_match)

                # append 2nd matchs to 1st matchs
                matched_detections = np.append(matched_detections, remaining_dets_to_match, axis = 0)
                matched_ids.extend(remaining_ids_to_match)
                classification_prob = np.append(classification_prob, remaining_class_prob, axis = 0)

            elif remaining_matrix.shape[0] == 1:
                """Case where only one detection is left."""
                # 2nd assignment problem (when one detection remaining)
                col_id_remainig = np.argmax(remaining_matrix, axis = 1)

                # 2nd ids, prob, and dets
                remaining_ids_to_match = ids_remaining.get(col_id_remainig[0])
                remaining_ids_to_match = [int(remaining_ids_to_match.split('_')[1])]

                remaining_class_prob = np.max(remaining_matrix)

                remaining_dets_to_match = remaining_detections
            
                # append 2nd matchs to 1st matchs
                matched_detections = np.append(matched_detections, remaining_dets_to_match, axis = 0)
                matched_ids.extend(remaining_ids_to_match)
                classification_prob = np.append(classification_prob, [remaining_class_prob], axis = 0)

        else:
            pass # do nothing otherwise
        

        # video output
        for _, (bbox, id) in enumerate(zip(matched_detections.tolist(), matched_ids)):
            prob = round(classification_prob[_], 2)

            text_position_1 = (int(bbox[0]), int(bbox[3]) - 10)
            text_position_2 = (int(bbox[0]) - 20, int(bbox[3]) + 40)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            font_color = palette[id] #(0, 255, 0)
            font_thickness = 3

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                          font_color, 3)

            cv2.putText(frame, f"cow_{id}", org = text_position_1, 
                        fontFace=font, fontScale = font_scale, color = font_color, 
                        thickness=font_thickness)
            
            cv2.putText(frame, f"prob {prob:.2f}", org = text_position_2, 
                        fontFace=font, fontScale = font_scale, color = font_color, 
                        thickness=font_thickness)

        # save txt file
        for (bbox, id) in zip(matched_detections.tolist(), matched_ids):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            f_output.write(f'{counter},{id},{x1},{y1},{x2},{y2}\n')

        # save image        
        #cv2.imwrite(f'eval/img/{video_input[:8]}frame_{counter}.png', frame)
        
        # write video
        cap_out.write(frame)

        # next frame
        ret, frame = cap.read()
        counter += 1

    cap.release()
    cap_out.release()
    cv2.destroyAllWindows()


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Custom")
    parser.add_argument("-i", "--video_input", help="Path to video in.", type=str)
    parser.add_argument("-o", "--video_output", help="Path to video out.", type=str)
    parser.add_argument("--output_file", help="Path to output file.", type=str)
    parser.add_argument("--thresh_1", help="Threshold 1.", type=float, default=0.9)
    parser.add_argument("--thresh_2", help="Threshold 2.", type=float, default=0.5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    Running the script.
    """
    print('Starting inference.')
    args = parse_args()
    main(args.video_input, args.video_output, args.output_file,
         args.thresh_1, args.thresh_2)
    print('Ending inference.')

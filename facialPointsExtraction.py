import cv2
import dlib
import numpy as np
from imutils import face_utils
import glob
import os
import csv

face_landmark_path = './shape_predictor_68_face_landmarks.dat'

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def main():
    # return

    

    path = 'srikiTest/*.mp4'
    files = glob.glob(path)
    

    for file in files:
        cap = cv2.VideoCapture(file)      
        points_file = open(file + '.csv', 'w')
        print(points_file)

        writer = csv.writer(points_file)
        writer.writerow(['Frame', 'X', 'Y'])

        #cap = cv2.VideoCapture('testVideos/sa1.mp4')
        if not cap.isOpened():
            print("Unable to connect to camera.")
            return
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(face_landmark_path)
        index = 0
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter('outputNew.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (600,600))
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                face_rects = detector(frame, 0)

                if len(face_rects) > 0:
                    shape = predictor(frame, face_rects[0])
                    shape = face_utils.shape_to_np(shape)

                    reprojectdst, euler_angle = get_head_pose(shape)

                    points = []

                    points.extend(shape[3:14]) #jaw position
                    points.extend(shape[48:55]) #upper outer lip
                    points.extend(shape[55:61]) #lower outer lip
                    points.extend(shape[61:65]) #inner upper lip
                    points.extend(shape[65:68]) #inner lower lip
                    points.extend(shape[31:36]) #nose position


                    for (x,y) in points:
                        #x = shape[i][0]
                        #y = shape[i][1]
                        cv2.putText(frame, ".", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), lineType=cv2.LINE_AA)
                        writer.writerow([index, x, y])
                    

                    index+=1
                    #writer.close()
                    '''
                    for i in range(48, 55):
                        x = shape[i][0]
                        y = shape[i][1]
                        cv2.putText(frame, "UOL", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255,255,255), lineType=cv2.LINE_AA)

                    for i in range(55, 61):
                        x = shape[i][0]
                        y = shape[i][1]
                        cv2.putText(frame, "LOL", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255,255,255), lineType=cv2.LINE_AA)

                    for i in range(61, 65):
                        x = shape[i][0]
                        y = shape[i][1]
                        cv2.putText(frame, "IUL", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255,255,255), lineType=cv2.LINE_AA)

                    for i in range(65, 68):
                        x = shape[i][0]
                        y = shape[i][1]
                        cv2.putText(frame, "ILL", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255,255,255), lineType=cv2.LINE_AA)

                    for i in range(31, 36):
                        x = shape[i][0]
                        y = shape[i][1]
                        cv2.putText(frame, "Nose", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255,255,255), lineType=cv2.LINE_AA)

                    '''
                    line_pairs = []
                    for start, end in line_pairs:
                        cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

                    cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 0), thickness=2)
                    cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 0), thickness=2)
                    cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 0), thickness=2)

                cv2.imshow("demo", frame)
                #print('number of frames completed', + frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        print('file completed', file)
        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



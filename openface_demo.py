import openface
import cv2
import numpy as np
import argparse
import os
import time

start = time.time()
np.set_printoptions(precision=2)

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))


def show_distances(dlib_face_predictor_path, torch_nn_model_path, img_dim=96):
    """Continuously compute and show distances between webcam frames"""

    cap = cv2.VideoCapture(0)

    # Set camera resolution

    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 360)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240)

    contrast = 1.0

    inner_start = time.time()
    align = openface.AlignDlib(dlib_face_predictor_path)
    net = openface.TorchNeuralNet(torch_nn_model_path, img_dim)
    if args.verbose:
        print("Loading the dlib and OpenFace models took {} seconds.".format(
            time.time() - inner_start))

    rep_last = None

    distances = []
    iteration = 0

    while(True):
        iteration += 1

        # Capture frame-by-frame
        ret, frame = cap.read()

        img = frame
        # Our operations on the frame come here
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Clamp the image
        img = np.minimum((img * contrast), 255.0).astype(np.uint8)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('+'):
            contrast = min(contrast + 0.1, 4.0)
            print('New contrast = ', contrast)
        elif key & 0xFF == ord('-'):
            contrast = max(0.0, contrast - 0.1)
            print('New contrast = ', contrast)

        inner_start = time.time()
        bb = align.getLargestFaceBoundingBox(img)
        # print(bb, type(bb))
        if bb is not None:
            face_detection_time = time.time() - inner_start

            # Draw the bbox
            cv2.rectangle(img, (bb.left(), bb.top()), (bb.right(), bb.bottom()), (255, 0, 0), 3)

            inner_start = time.time()
            aligned_face = align.align(img_dim, img, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

            if aligned_face is not None:
                face_alignment_time = time.time() - inner_start

                inner_start = time.time()
                rep_new = net.forward(aligned_face)

                if rep_last is None:
                    rep_last = rep_new

                d = rep_new - rep_last
                d = np.dot(d, d)
                rep_last = rep_new
                distances.append(d)

                computation_time = time.time() - inner_start

                print("{}\tDistance: {}".format(iteration, d))
                print("\tFace detected in:\t{:0.4f}s".format(face_detection_time))
                print("\tFace aligned in:\t{:0.4f}s".format(face_alignment_time))
                print("\tFace computed in:\t{:0.4f}s".format(computation_time))
            else:
                print("{}\tFace not aligned".format(iteration))
                print("\tFace detected in:\t{:0.4f}s".format(face_detection_time))
        else:
            print("{}\tNo face detected".format(iteration))

        # Display the resulting frame
        cv2.imshow('frame', img)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    print("Distances mean: {}".format(sum(distances) / len(distances)))


if __name__ == '__main__':
    show_distances(args.dlibFacePredictor, args.networkModel, args.imgDim)
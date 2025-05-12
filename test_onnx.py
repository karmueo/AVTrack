import torch
import onnxruntime
import cv2
from lib.train.data.processing_utils import sample_target
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
import numpy as np


save_name = 'avtrack.onnx'
image_path = "test.jpg"
info = {'init_bbox': [2, 724, 302, 554]}
image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
H, W, _ = image.shape
template_factor = 2.0
search_factor = 4.0
template_size = 128
search_size = 256
g_state = [0, 0, 0, 0]


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def map_box_back(state, search_size, pred_box: list, resize_factor: float):
    cx_prev, cy_prev = state[0] + 0.5 * \
        state[2], state[1] + 0.5 * state[3]
    cx, cy, w, h = pred_box
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


class AVTrack():
    def __init__(self, onnx_model_path):
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)
        self.preprocessor = Preprocessor()
        self.state = None
        self.template_factor = template_factor
        self.search_factor = search_factor
        self.template_size = template_size
        self.search_size = search_size

        # for debug
        self.frame_id = 0
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.template_factor,
                                                                output_sz=self.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        self.z_dict1 = template

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.search_factor,
                                                                # (x1, y1, w, h)
                                                                output_sz=self.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        template_arr = to_numpy(self.z_dict1.tensors)
        search_arr = to_numpy(search.tensors)
        """ f = open("template_arr.txt", 'w')
        np.savetxt(f, template_arr.flatten())
        f.close()

        f = open("search_arr.txt", 'w')
        np.savetxt(f, search_arr.flatten())
        f.close() """

        ort_inputs = {'template': template_arr,
                      'search': search_arr}

        ort_outs = self.ort_session.run(None, ort_inputs)

        tensor = torch.from_numpy(ort_outs[0])
        pred_boxes = tensor.view(-1, 4)

        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            # (cx, cy, w, h) [0,1]
            dim=0) * self.search_size / resize_factor).tolist()
        # get the final box result
        self.state = clip_box(self.map_box_back(
            pred_box, resize_factor), H, W, margin=10)
        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * \
            self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def main(videofilepath):
    def _build_init_info(box):
        return {'init_bbox': box}

    cap = cv2.VideoCapture(videofilepath)
    display_name = 'Display: onnx'
    cv2.namedWindow(display_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(display_name, 960, 720)
    success, frame = cap.read()
    cv2.imshow(display_name, frame)

    tracker = AVTrack(save_name)

    while True:
        # cv.waitKey()
        frame_disp = frame.copy()

        cv2.putText(frame_disp,
                    'Select target ROI and press ENTER',
                    (20, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1.5, (0, 0, 0), 1)

        x, y, w, h = cv2.selectROI(
            display_name, frame_disp, fromCenter=False)
        init_state = [x, y, w, h]
        tracker.initialize(frame, _build_init_info(init_state))
        break

    while True:
        ret, frame = cap.read()

        if frame is None:
            break

        frame_disp = frame.copy()

        # Draw box
        out = tracker.track(frame)
        state = [int(s) for s in out['target_bbox']]

        cv2.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                      (0, 255, 0), 1)

        font_color = (0, 0, 0)
        cv2.putText(frame_disp, 'Tracking!', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    font_color, 1)
        cv2.putText(frame_disp, 'Press r to reset', (20, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    font_color, 1)
        cv2.putText(frame_disp, 'Press q to quit', (20, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    font_color, 1)

        # Display the resulting frame
        cv2.imshow(display_name, frame_disp)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


if __name__ == '__main__':
    main("/home/tl/data/ball/vidoes/2025-05-09_173338_104.mp4")

""" z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'],
                                                        template_factor,
                                                        output_sz=template_size)
cv2.imwrite("z_patch_arr.jpg", z_patch_arr)
preprocessor = Preprocessor()
template = preprocessor.process(z_patch_arr, z_amask_arr)

x_patch_arr, resize_factor, x_amask_arr = sample_target(image,
                                                        info['init_bbox'],
                                                        search_factor,
                                                        output_sz=search_size)
cv2.imwrite("x_patch_arr.jpg", x_patch_arr)
search = preprocessor.process(x_patch_arr, x_amask_arr)

ort_session = onnxruntime.InferenceSession(save_name)

template_arr = to_numpy(template.tensors)
search_arr = to_numpy(search.tensors)
f = open("template_arr.txt", 'w')
np.savetxt(f, template_arr.flatten())
f.close()

f = open("search_arr.txt", 'w')
np.savetxt(f, search_arr.flatten())
f.close()

ort_inputs = {'template': template_arr,
              'search': search_arr}

ort_outs = ort_session.run(None, ort_inputs)

tensor = torch.from_numpy(ort_outs[0])
pred_boxes = tensor.view(-1, 4)
print("pred_boxes:", pred_boxes)

pred_box = (pred_boxes.mean(
            # (cx, cy, w, h) [0,1]
            dim=0) * search_size / resize_factor).tolist()

state = clip_box(map_box_back(
    info['init_bbox'], search_size, pred_box, resize_factor), H, W, margin=10)

print("Predicted state:", state) """

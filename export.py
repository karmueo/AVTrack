from lib.models.avtrack import build_avtrack
import os
import sys
import argparse
import importlib
import torch
import onnx
import onnxruntime

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, template, search, template_anno, search_anno, is_distill):
        outputs = self.model(
            template, search, template_anno, search_anno, is_distill)
        # 假设 pred_boxes, score_map, size_map 分别是 outputs 的前3个
        pred_boxes = outputs['pred_boxes']
        score_map = outputs['score_map']
        size_map = outputs['size_map']
        return pred_boxes  # , score_map, size_map


def get_data():
    img_x = torch.randn(1, 3, 256, 256, requires_grad=True)
    feat_vec_z = torch.randn(1, 3, 128, 128, requires_grad=True)
    return feat_vec_z, img_x


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str,
                        help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str,
                        help='Name of parameter file.')

    args = parser.parse_args()

    param_module = importlib.import_module(
        'lib.test.parameter.{}'.format(args.tracker_name))
    params = param_module.parameters(args.tracker_param)

    network = build_avtrack(params.cfg, training=False)
    network.load_state_dict(torch.load(
        params.checkpoint, map_location='cpu', weights_only=False)['net'],
        strict=True)
    network.eval()
    # print(network)

    export_model = ExportWrapper(network)

    feat_vec_z, img_x = get_data()

    torch_outs = export_model(
        feat_vec_z,
        img_x,
        template_anno=[],
        search_anno=[],
        is_distill=params.cfg.MODEL['IS_DISTILL'])

    print("torch outputs:")
    print(torch_outs)

    model_input = (feat_vec_z,
                   img_x,
                   [],
                   [],
                   params.cfg.MODEL['IS_DISTILL'])

    save_name = "avtrack.onnx"

    torch.onnx.export(export_model,  # model being run
                      # model input (or a tuple for multiple inputs)
                      model_input,
                      # where to save the model (can be a file or file-like object)
                      save_name,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=14,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      # the model's input names
                      input_names=['template', 'search',
                                   "template_anno", "search_anno"],
                      # the model's output names
                      output_names=['pred_boxes'],
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                      #               'output': {0: 'batch_size'}}
                      )

    onnx_model = onnx.load(save_name)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(save_name)

    ort_inputs = {'template': to_numpy(feat_vec_z),
                  'search': to_numpy(img_x)}

    ort_outs = ort_session.run(None, ort_inputs)
    print("onnx outputs:")
    print(ort_outs)


if __name__ == "__main__":
    main()

import sys
import torch
from tool.darknet2pytorch import Darknet

def convert_to_onnx(cfgfile, weightfile, img_size=608, batch_size=1):
    model = Darknet(cfgfile, img_size)
    model.load_weights(weightfile)
    model.eval()
    print("Model loaded with weights.")

    dummy_input = torch.randn(batch_size, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy_input, f"yolov4_{img_size}.onnx",
        verbose=False, opset_version=11,
        input_names=["input"], output_names=["output"]
    )
    print(f"ONNX export done → yolov4_{img_size}.onnx")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python convert_to_onnx.py <cfgfile> <weightfile> <img_size>")
        sys.exit(1)

    cfgfile = sys.argv[1]
    weightfile = sys.argv[2]
    img_size = int(sys.argv[3])
    convert_to_onnx(cfgfile, weightfile, img_size)

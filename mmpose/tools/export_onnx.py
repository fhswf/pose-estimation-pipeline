import argparse
import os
import sys
# Add current directory to path to allow importing mmpose
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from mmpose.apis import init_model
from mmengine.registry import init_default_scope

class ModelWrapper(nn.Module):
    """
    Wrapper class to expose the raw head outputs (SimCC representations) for ONNX export.
    This bypasses the standard MMPose post-processing which involves complex decoding
    not suitable for direct ONNX export without MMDeploy plugins.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Extract features from backbone (and neck if present)
        feat = self.model.extract_feat(x)
        # Pass features through the head to get raw SimCC outputs (pred_x, pred_y)
        # Note: We call forward() directly on the head to skip the 'predict' logic that returns datasamples
        return self.model.head.forward(feat)

def parse_args():
    parser = argparse.ArgumentParser(description='Export MMPose model to ONNX')
    parser.add_argument('config', help='Path to the config file')
    parser.add_argument('checkpoint', help='Path to the checkpoint file')
    parser.add_argument('--output', default='model.onnx', help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--device', default='cpu', help='Device to use for export')
    parser.add_argument(
        '--input-shape',
        default=None,
        nargs=2,
        type=int,
        help='Input shape (batch, channel, height, width) will be inferred from config if not specified. '
             'Example: --input-shape 288 384'
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize default scope to avoid registry errors
    init_default_scope('mmpose')

    print(f"Loading model from {args.config}...")
    # Load the model
    model = init_model(args.config, args.checkpoint, device=args.device)
    model.eval()

    # Wrap the model
    wrapped_model = ModelWrapper(model)

    # Determine input shape
    if args.input_shape:
        # User specified H W, assuming defaults for B C
        input_h, input_w = args.input_shape
    else:
        # Try to read from codec config
        try:
            # Look for input_size in codec/data_preprocessor/dataset config
            # configs/gruppe1.py has input_size = (288, 384) defined at top level
            # but usually it ends up in codec or model.test_cfg
            if hasattr(model.cfg, 'codec') and 'input_size' in model.cfg.codec:
                input_size = model.cfg.codec['input_size']
                # input_size in config is usually (W, H)
                input_w, input_h = input_size
            elif hasattr(model.cfg, 'input_size'):
                 input_w, input_h = model.cfg.input_size
            else:
                 # Fallback to standard 256x192 if not found
                 print("Warning: Could not auto-detect input size, using default 192x256 (W,H). Use --input-shape H W if incorrect.")
                 input_w, input_h = 192, 256
        except Exception as e:
            print(f"Warning: Error detecting input size: {e}. Using default 192x256.")
            input_w, input_h = 192, 256
    
    print(f"Using input shape: 1x3x{input_h}x{input_w} (BxCxHxW)")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_h, input_w).to(args.device)

    # dynamic axes for variable batch size
    dynamic_axes = {
        'input': {0: 'batch'},
        'simcc_x': {0: 'batch'},
        'simcc_y': {0: 'batch'}
    }

    print(f"Exporting to {args.output}...")
    try:
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            args.output,
            input_names=['input'],
            output_names=['simcc_x', 'simcc_y'],
            export_params=True,
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
            verbose=False,
            opset_version=args.opset,
            dynamic_axes=dynamic_axes
        )
        print(f"Successfully exported ONNX model to {args.output}")
        
        # Verify if file exists
        if os.path.exists(args.output):
             print(f"File size: {os.path.getsize(args.output) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"Export failed: {e}")
        exit(1)

if __name__ == '__main__':
    main()

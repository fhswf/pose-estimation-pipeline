import torch
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Simple3DPoseLiftingModel import Simple3DPoseLiftingModel

def export_model(num_keypoints, model_path, output_path=None):
    if output_path is None:
        output_path = model_path.replace('.pth', '.onnx')
    
    print(f"Loading model from {model_path} for {num_keypoints} keypoints...")
    model = Simple3DPoseLiftingModel(num_keypoints=num_keypoints)
    
    # Load weights
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, num_keypoints * 2)
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    # In newer PyTorch versions, `torch.onnx.export` might save to dual `.onnx` and `.data` files.
    # We consolidate them using the `onnx` library.
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        # Re-save with save_as_external_data=False to pack the weights inline
        onnx.save_model(onnx_model, output_path, save_as_external_data=False)
        
        # Cleanup the unneeded external .data file if PyTorch created it
        data_file_path = f"{output_path}.data"
        if os.path.exists(data_file_path):
            os.remove(data_file_path)
    except ImportError:
        print("Warning: `onnx` library not found. Assuming output defaults kept...")

    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export lifting model to ONNX")
    parser.add_argument("--num_keypoints", type=int, choices=[17, 133], required=True, help="Number of keypoints")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth model file")
    parser.add_argument("--output_path", type=str, help="Optional output path for .onnx file")
    
    args = parser.parse_args()
    export_model(args.num_keypoints, args.model_path, args.output_path)

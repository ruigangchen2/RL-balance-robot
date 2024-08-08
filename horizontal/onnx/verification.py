import onnx
import onnxruntime

# Load the ONNX model
model = onnx.load("PPO.onnx")
# Check that the IR is well formed
onnx.checker.check_model(model)
# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))


session = onnxruntime.InferenceSession('PPO.onnx', None)
raw_result = session.run([], {"onnx::Gemm_0": [[90, 200, 110]]})
print(raw_result)
import onnxruntime as rt
import numpy as  np
data = np.array([[-90,0,-1800]])
sess = rt.InferenceSession('PPO.onnx')
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

pred_onx = sess.run([label_name], {input_name:data.astype(np.float32)})[0]
print(pred_onx)
print(np.argmax(pred_onx))

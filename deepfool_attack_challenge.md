# DeepFool Challenge

## Objective



Your task is to craft a targeted adversarial example using a
DeepFool-style iterative attack. Unlike untargeted attacks that simply
cause any misclassification, this challenge requires you to fool the
classifier into predicting a specific target class while minimizing the
perturbation under anℓ2\ell_2distance constraint.





The constraint is‖xadv−x‖2≤threshold\lVert x_{\text{adv}} - x \rVert_2 \leq \text{threshold},
wherexxis the baseline image andxadvx_{\text{adv}}is your crafted adversarial example. Both conditions must hold: the
predicted class must equal the specified target label (not just any
misclassification), and theℓ2\ell_2distance must be at most the threshold.



## Quick Start

All API endpoints expect images in`[0,1]`pixel space, not normalized tensors.

Start the instance from the platform. Check readiness:

Code:bash```
`exportBASE_URL="http://instance_ip:port"curl-s"$BASE_URL/health"`
```

## API

The API is minimal and deterministic. Each response is JSON. Images are base64-encoded`PNG`files of shape`28x28`and single channel in`[0,1]`after decoding.

### GET /health

Returns service status and configuration parameters.

Code:bash```
`curl-s"$BASE_URL/health"|jq`
```

Example response:

Code:json```
`{"status":"ok","l2_threshold":0.75,"index":95,"target":6}`
```



The`l2_threshold`defines the maximumL2L_2distance allowed. The`index`identifies the MNIST test
sample. The`target`specifies the required target class for
the adversarial example.



### GET /challenge

Returns the fixed sample and the constraint that must be satisfied.

Code:bash```
`curl-s"$BASE_URL/challenge"|jq`
```

Example response shape:

Code:json```
`{"sample_index":95,"label":4,"target":6,"l2_threshold":0.75,"image_b64":"<base64 PNG>"}`
```

The`sample_index`identifies the specific MNIST test sample used for this challenge. The`label`is the ground-truth class of the baseline image. The`target`is the required predicted class for a successful adversarial example (targeted attack). The`l2_threshold`defines the maximum $\ell_2$ distance allowed in`[0,1]`pixel space. The`image_b64`field contains the baseline image as a base64-encoded`PNG`.

### POST /predict

Returns the model's prediction for an arbitrary image. Use this endpoint to validate the pipeline, encoding, and intermediate candidates. It does not return the flag.

Code:bash```
`curl-s -X POST"$BASE_URL/predict"\-H'content-type: application/json'\-d'{"image_b64": "<base64 PNG of 28x28 grayscale>"}'|jq`
```

Example response shape:

Code:json```
`{"pred":8,"confidence":0.73}`
```

The`pred`field contains the predicted class (integer in`[0,9]`). The`confidence`field provides the softmax probability for the predicted class (float in`[0,1]`).

### GET /weights

Returns the pretrained model`state_dict`as a binary file. Load it on CPU and use the provided classifier architecture to compute input gradients locally.

Code:bash```
`curl-s -o deepfool_weights.pth"$BASE_URL/weights"`
```

The downloaded file is a PyTorch state dictionary compatible with the`SimpleClassifier`architecture detailed below. Load with`torch.load(path, map_location=torch.device('cpu'))`.

### POST /submit



Validates that the candidate is a proper`28x28`single-channel image in`[0,1]`, that theℓ2\ell_2distance to the baseline image is at most`l2_threshold`, and
that the classifier’s predicted class matches the target label. Returns
the flag on success.



Code:bash```
`curl-s -X POST"$BASE_URL/submit"\-H'content-type: application/json'\-d'{"image_b64": "<base64 PNG of 28x28 grayscale>"}'|jq`
```

Example success response shape:

Code:json```
`{"ok":true,"pred":6,"target":6,"l2":0.68,"flag":"HTB{...}"}`
```

On validation failure, the endpoint returns HTTP 400 with a descriptive error message (e.g.,`"L2 too large: 0.82 > 0.75"`or`"Wrong target: predicted 8, need 6"`).

## Minimal Python Scaffolds

The following scaffolds provide API helpers, model architecture, and integration testing. They do not implement the attack.

### API Helpers and Challenge Fetching

Code:python```
`importos,io,base64,numpyasnp,requests,torch,torch.nnasnnfromPILimportImage

BASE_URL=os.getenv("BASE_URL","http://127.0.0.1:8000")MNIST_MEAN,MNIST_STD=0.1307,0.3081defx01_from_b64_png(b64:str)->np.ndarray:"""Convert base64 PNG to [0,1] numpy array.

    Args:
        b64: Base64 encoded PNG string

    Returns:
        np.ndarray: Image as (28, 28) array in [0,1] range
    """raw=base64.b64decode(b64)img=Image.open(io.BytesIO(raw)).convert("L")ifimg.size!=(28,28):raiseValueError("Expected 28x28 PNG")x=np.asarray(img,dtype=np.float32)/255.0returnnp.clip(x,0.0,1.0)defb64_png_from_x01(x2d:np.ndarray)->str:"""Convert [0,1] array to base64 PNG.

    Args:
        x2d: Image array in [0,1] range

    Returns:
        str: Base64 encoded PNG string
    """x255=np.clip((x2d*255.0).round(),0,255).astype(np.uint8)img=Image.fromarray(x255,mode="L")buf=io.BytesIO()img.save(buf,format="PNG",optimize=True)returnbase64.b64encode(buf.getvalue()).decode("ascii")defl2(a:np.ndarray,b:np.ndarray)->float:"""Compute L2 distance between two arrays.

    Args:
        a, b: Arrays to compare

    Returns:
        float: Euclidean distance
    """returnfloat(np.linalg.norm((a-b).ravel(),ord=2))# Fetch challenge and check clean predictionch=requests.get(f"{BASE_URL}/challenge",timeout=10).json()x=x01_from_b64_png(ch["image_b64"])# (28, 28)lab=int(ch["label"])# baseline labeltgt=int(ch["target"])# target labelthr=float(ch["l2_threshold"])# numeric thresholdres=requests.post(f"{BASE_URL}/predict",json={"image_b64":b64_png_from_x01(x)},timeout=10).json()print({"baseline_label":lab,"target":tgt,"server_pred":res["pred"],"l2_threshold":thr})`
```

### Model Architecture and Loading

The server uses this architecture. You need to replicate it locally to compute gradients.

Code:python```
`classSimpleClassifier(nn.Module):def__init__(self):super().__init__()self.conv1=nn.Conv2d(1,32,3,1)self.conv2=nn.Conv2d(32,64,3,1)self.dropout1=nn.Dropout(0.25)self.dropout2=nn.Dropout(0.5)self.fc1=nn.Linear(9216,128)self.fc2=nn.Linear(128,10)defforward(self,x01:torch.Tensor)->torch.Tensor:"""Forward pass with internal normalization.

        Args:
            x01: Input tensor in [0,1] with shape (N, 1, 28, 28)

        Returns:
            Log-probabilities with shape (N, 10)
        """x=(x01-MNIST_MEAN)/MNIST_STD
        x=torch.relu(self.conv1(x))x=torch.relu(self.conv2(x))x=torch.max_pool2d(x,2)x=self.dropout1(x)x=torch.flatten(x,1)x=torch.relu(self.fc1(x))x=self.dropout2(x)x=self.fc2(x)returntorch.log_softmax(x,dim=1)# Download and load weightswt=requests.get(f"{BASE_URL}/weights",timeout=10).contentopen("deepfool_weights.pth","wb").write(wt)model=SimpleClassifier().eval()state=torch.load("deepfool_weights.pth",map_location=torch.device("cpu"))model.load_state_dict(state)# Verify model works locallyx_tensor=torch.from_numpy(x[None,None,...]).float()logits=model(x_tensor)local_pred=int(torch.argmax(logits,dim=1).item())print(f"Local prediction:{local_pred}, should match server:{res['pred']}")`
```

### Testing Server Validation

Verify server-side checks by submitting the clean image and observing expected failures.

Code:python```
`bad=requests.post(f"{BASE_URL}/submit",json={"image_b64":b64_png_from_x01(x)},timeout=10)print(bad.status_code,bad.text)# expected: 400 with "Wrong target: predicted X, need 6"`
```
# Jacobian-based Saliency Map Attack Challenge

Your task is to craft a targeted adversarial example that fools an MNIST digit classifier using the Jacobian-based Saliency Map Attack (JSMA). You will receive a baseline image that the classifier correctly predicts. Your job is to modify a small number of pixels to cause the classifier to predict a specific target class, demonstrating precise, sparse adversarial perturbations.

What you need to do:

Fetch the challenge from the API to receive a baseline`MNIST`image, its ground-truth label, a target class, and an`L0`budget (maximum number of pixels you can modify). Use JSMA to iteratively select and modify the most salient pixels that increase the target class score while decreasing other class scores. Submit your adversarial image to the API. If the classifier predicts the target class and you modified at most the allowed number of pixels, you receive the flag.

The mathematical constraint is an $L_0$ bound on the number of modified pixels:



‖xadv−x‖0≤budget\lVert x_{\text{adv}} - x \rVert_0 \leq \text{budget}





wherexxis the baseline image,xadvx_{\text{adv}}is your crafted adversarial example, and‖⋅‖0\lVert \cdot \rVert_0counts the number of pixels with non-zero differences (differences above
a small threshold like10−610^{-6}).
The classifier must predict the specified target classtt,
and all pixel values must remain in[0,1][0,1].



## Quick Start

All API endpoints expect images in`[0,1]`pixel space, not normalized tensors.

Start the instance from the platform. Check readiness:

```
icantthinkofaname23@htb[/htb]`$exportBASE_URL="http://instance_ip:port"`
```

```
icantthinkofaname23@htb[/htb]`$curl-s"$BASE_URL/health"{"status":"healthy"}`
```

## API

The API is minimal and deterministic. Each response is JSON. Images are base64-encoded`PNG`files of shape`28x28`and single channel in`[0,1]`after decoding.

### GET /health

Returns service status and configuration parameters.

```
icantthinkofaname23@htb[/htb]`$curl-s"$BASE_URL/health"|jq{
  "status": "healthy"
}`
```

### GET /challenge

Returns the fixed sample and attack parameters that must be satisfied.

```
icantthinkofaname23@htb[/htb]`$curl-s"$BASE_URL/challenge"|jq{
  "sample_index": 2,
  "target_class": 7,
  "l0_budget": 50,
  "original_label": 1,
  "max_l2": 8.0,
  "image_b64": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAiUlEQVR42mNgoD9gRhdQe/3uFIzNhC5p+O8pbqPaP+KW0/0yDcFBN1adayVunafuc+OUU/h3gwGnsfYMr3FL6jJ04TTV8u1ZDpw6nYVu/MApqf9/DU5TJV5cR+Gj6EwQO4FbUp7hPe7gefLPCadOW3E8Ed/77ywzLp1cXgxr/uLSyHpsAxfDgAMAUBMg+xetJ/UAAAAASUVORK5CYII="
}`
```



The`sample_index`identifies the specific MNIST test
sample used for this challenge. The`original_label`is the
ground-truth class of the baseline image. The`target_class`is the specific class the adversarial example must be classified as. The`l0_budget`defines the maximum number of pixels that can be
modified. The`max_l2`parameter serves as a safeguard to
prevent replacing the entire image with unrelated data. The`image_b64`field contains the baseline image as a
base64-encoded`PNG`.



### POST /predict

Returns the model's prediction for an arbitrary image. Use this endpoint to validate the pipeline, encoding, and intermediate candidates. It does not return the flag.

```
icantthinkofaname23@htb[/htb]`$curl-s -X POST"$BASE_URL/predict"\
  -H 'content-type: application/json' \
  -d '{"image_b64": "<base64 PNG of 28x28 grayscale>"}' | jq
{
  "predicted_class": 7,
  "confidence": 0.89,
  "all_probabilities": [0.01, 0.02, 0.01, 0.03, 0.01, 0.02, 0.01, 0.89, 0.00, 0.00]
}`
```

The`predicted_class`field contains the predicted class (integer in`[0,9]`). The`confidence`field provides the probability for the predicted class (float in`[0,1]`). The`all_probabilities`array contains the probability distribution over all 10 classes.

### GET /weights

Returns the pretrained model`state_dict`as a binary file. Load it on CPU and use the provided classifier architecture to compute the Jacobian matrix locally.

```
icantthinkofaname23@htb[/htb]`$curl-s -o jsma_weights.pth"$BASE_URL/weights"`
```

The downloaded file is a PyTorch state dictionary compatible with the`MNISTClassifier`(LeNet-5 style) architecture detailed below. Load with`torch.load(path, map_location=torch.device('cpu'))`.

### POST /submit



Validates that the candidate is a proper`28x28`single-channel image in[0,1][0,1],
that the number of modified pixels is at most`l0_budget`,
that theL2L_2distance is below`max_l2`, that the classifier’s predicted
class equals the target class, and that the submission is not simply a
clean image of the target class copied from the dataset. Returns the
flag on success.



```
icantthinkofaname23@htb[/htb]`$curl-s -X POST"$BASE_URL/submit"\
  -H 'content-type: application/json' \
  -d '{"image_b64": "<base64 PNG of 28x28 grayscale>"}' | jq
{
  "success": true,
  "flag": "HTB{...}",
  "pixels_modified": 45,
  "predicted_class": 7,
  "target_class": 7
}`
```

On validation failure, the endpoint returns HTTP 400 with a descriptive error message (e.g.,`"L0 constraint violated: 55 pixels modified (max: 50)"`,`"Target misclassification failed: predicted 3, expected 7"`, or`"Submission rejected: please derive the adversarial example from the provided baseline image"`).

## Minimal Python Scaffolds

The following scaffolds provide API helpers, model architecture, and integration testing. They do not implement the attack.

### API Helpers and Challenge Fetching

Code:python```
`importos,io,base64,numpyasnp,requests,torch,torch.nnasnn,torch.nn.functionalasFfromPILimportImage

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
    """x255=np.clip((x2d*255.0).round(),0,255).astype(np.uint8)img=Image.fromarray(x255,mode="L")buf=io.BytesIO()img.save(buf,format="PNG",optimize=True)returnbase64.b64encode(buf.getvalue()).decode("ascii")defcount_modified_pixels(a:np.ndarray,b:np.ndarray,threshold:float=1e-6)->int:"""Count number of modified pixels (L0 norm).

    Args:
        a, b: Arrays to compare
        threshold: Minimum difference to count as modified

    Returns:
        int: Number of pixels with absolute difference above threshold
    """returnint(np.sum(np.abs(a-b)>threshold))# Fetch challenge and check clean predictionch=requests.get(f"{BASE_URL}/challenge",timeout=10).json()x=x01_from_b64_png(ch["image_b64"])# (28, 28)orig_label=int(ch["original_label"])# baseline labeltarget=int(ch["target_class"])# target class for attackbudget=int(ch["l0_budget"])# max pixels to modifymax_l2=float(ch["max_l2"])# L2 safeguardres=requests.post(f"{BASE_URL}/predict",json={"image_b64":b64_png_from_x01(x)},timeout=10).json()print({"original_label":orig_label,"server_pred":res["predicted_class"],"target_class":target,"l0_budget":budget,"max_l2":max_l2})`
```

### Model Architecture and Loading

The server uses a LeNet-5 style architecture. You need to replicate it locally to compute the Jacobian matrix.

Code:python```
`classMNISTClassifier(nn.Module):"""LeNet-5 style classifier.

    Architecture:
    - Conv1: 1 -> 6 channels, 5x5 kernel, Tanh activation, 2x2 avg pooling
    - Conv2: 6 -> 16 channels, 5x5 kernel, Tanh activation, 2x2 avg pooling
    - FC1: 256 -> 120, Tanh activation
    - FC2: 120 -> 84, Tanh activation
    - FC3: 84 -> 10, log-softmax output
    """def__init__(self):super().__init__()self.conv1=nn.Conv2d(1,6,kernel_size=5,stride=1,padding=0)self.conv2=nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0)self.pool=nn.AvgPool2d(kernel_size=2,stride=2)self.fc1=nn.Linear(16*4*4,120)self.fc2=nn.Linear(120,84)self.fc3=nn.Linear(84,10)self.act=nn.Tanh()defforward(self,x:torch.Tensor)->torch.Tensor:"""Forward pass returning log-softmax outputs.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28) in [0,1] range

        Returns:
            Log-softmax output of shape (batch, 10)
        """x=self.act(self.conv1(x))# (B,6,24,24)x=self.pool(x)# (B,6,12,12)x=self.act(self.conv2(x))# (B,16,8,8)x=self.pool(x)# (B,16,4,4)x=torch.flatten(x,1)# (B,256)x=self.act(self.fc1(x))# (B,120)x=self.act(self.fc2(x))# (B,84)x=self.fc3(x)# (B,10)returnF.log_softmax(x,dim=1)defmnist_normalize(x01:torch.Tensor)->torch.Tensor:"""Normalize [0,1] tensor to MNIST statistics.

    Args:
        x01: Input tensor in [0,1] range

    Returns:
        Normalized tensor
    """return(x01-MNIST_MEAN)/MNIST_STD# Download and load weightswt=requests.get(f"{BASE_URL}/weights",timeout=10).contentopen("jsma_weights.pth","wb").write(wt)model=MNISTClassifier().eval()state=torch.load("jsma_weights.pth",map_location=torch.device("cpu"))model.load_state_dict(state)# Verify model works locallyx_tensor=torch.from_numpy(x[None,None,...]).float()logits=model(mnist_normalize(x_tensor))local_pred=int(torch.argmax(logits,dim=1).item())print(f"Local prediction:{local_pred}, should match server:{res['predicted_class']}")`
```

### Testing Server Validation

Verify server-side checks by submitting the clean image and observing expected failures.

Code:python```
`bad=requests.post(f"{BASE_URL}/submit",json={"image_b64":b64_png_from_x01(x)},timeout=10)print(bad.status_code,bad.text)# expected: 400 with "Target misclassification failed"`
```
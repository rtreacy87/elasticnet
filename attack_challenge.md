# ElasticNet Attack Challenge

Your task is to craft an adversarial example that fools an MNIST digit classifier using the ElasticNet (EAD) attack. You will receive a baseline image that the classifier correctly predicts. Your job is to add a carefully crafted perturbation that causes misclassification while respecting multiple distance constraints simultaneously.

What you need to do:

Fetch the challenge from the API to receive a baseline`MNIST`image, its ground-truth label, and the constraint parameters. Use the ElasticNet attack (FISTA optimization with elastic-net regularization and binary search) to modify the image. Ensure your perturbation satisfies all three distance bounds: the elastic-net distance (combining`L2`and`L1`norms), an`L2`cap, and an`L1`cap. Submit your adversarial image to the API. If the classifier misclassifies it and all constraints are satisfied, you receive the flag.



To craft a valid adversarial example,xadvx_{\text{adv}},
from a baseline image,xx,
several mathematical constraints must be met simultaneously. The
perturbation is bounded by three conditions: an`elastic-net constraint`(‖xadv−x‖2+β‖xadv−x‖1≤elastic_max\lVert x_{\text{adv}} - x \rVert_2 + \beta \lVert x_{\text{adv}} - x \rVert_1 \leq \text{elastic\\_max}),
an`auxiliary L2 norm constraint`(‖xadv−x‖2≤l2_max\lVert x_{\text{adv}} - x \rVert_2 \leq \text{l2\\_max}),
and an`auxiliary L1 norm constraint`(‖xadv−x‖1≤l1_max\lVert x_{\text{adv}} - x \rVert_1 \leq \text{l1\\_max}),
whereβ\betais the elastic-net trade-off parameter. Furthermore, for the attack to
be successful, the predicted class of the adversarial example must
differ from the baseline label, and all its pixel values must remain
valid by being clipped within the range[0,1][0,1].



## Quick Start

All API endpoints expect images in`[0,1]`pixel space, not normalized tensors.

Start the instance from the platform. Check readiness:

```
icantthinkofaname23@htb[/htb]`$exportBASE_URL="http://instance_ip:port"`
```

```
icantthinkofaname23@htb[/htb]`$curl-s"$BASE_URL/health"{"status":"ok"}`
```

## API

The API is minimal and deterministic. Each response is JSON. Images are base64-encoded`PNG`files of shape`28x28`and single channel in`[0,1]`after decoding.

### GET /health

Returns service status and configuration parameters.

```
icantthinkofaname23@htb[/htb]`$curl-s"$BASE_URL/health"|jq{
  "status": "ok"
}`
```

### GET /challenge

Returns the fixed sample and all constraints that must be satisfied.

```
icantthinkofaname23@htb[/htb]`$curl-s"$BASE_URL/challenge"|jq{
  "sample_index": ...,
  "label": ...,
  "beta": ...,
  "elastic_max": ...,
  "l2_max": ...,
  "l1_max": ...,
  "image_b64": "..."
}`
```



The`sample_index`identifies the specific MNIST test
sample used for this challenge. The`label`is the
ground-truth class. The`beta`parameter controls the
elastic-net trade-off betweenL2L_2andL1L_1regularization. The three distance bounds (`elastic_max`,`l2_max`,`l1_max`) define the maximum allowed
distortion in`[0,1]`pixel space. The`image_b64`field contains the baseline image as a base64-encoded`PNG`.



### POST /predict

Returns the model's prediction for an arbitrary image. Use this endpoint to validate the pipeline, encoding, and intermediate candidates. It does not return the flag.

```
icantthinkofaname23@htb[/htb]`$curl-s -X POST"$BASE_URL/predict"\
  -H 'content-type: application/json' \
  -d '{"image_b64": "<base64 PNG of 28x28 grayscale>"}' | jq
{
  "pred": ...,
  "confidence": ...
}`
```

The`pred`field contains the predicted class (integer in`[0,9]`). The`confidence`field provides the softmax probability for the predicted class (float in`[0,1]`).

### GET /weights

Returns the pretrained model`state_dict`as a binary file. Load it on CPU and use the provided classifier architecture to compute input gradients locally.

```
icantthinkofaname23@htb[/htb]`$curl-s -o elasticnet_weights.pth"$BASE_URL/weights"`
```

The downloaded file is a PyTorch state dictionary compatible with the`SimpleClassifier`architecture detailed below. Load with`torch.load(path, map_location=torch.device('cpu'))`.

### POST /submit

Validates that the candidate is a proper`28x28`



single-channel image in`[0,1]`, that all three distance
constraints are satisfied (elastic-net distance,L2L_2bound, andL1L_1bound), and that the classifier’s predicted class differs from the
baseline label. Returns the flag on success.





```
icantthinkofaname23@htb[/htb]`$curl-s -X POST"$BASE_URL/submit"\
  -H 'content-type: application/json' \
  -d '{"image_b64": "<base64 PNG of 28x28 grayscale>"}' | jq
{
  "ok": true,
  "flag": "HTB{...}",
  "pred": ...,
  "metrics": {
    "l1": ...,
    "l2": ...,
    "linf": ...,
    "elastic": ...
  }
}`
```

On validation failure, the endpoint returns HTTP 400 with a descriptive error message (e.g.,`"Elastic too large: 1.85 > 1.80"`,`"L2 too large: 1.42 > 1.40"`, or`"Not misclassified; try stronger within bounds"`).

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
    """x255=np.clip((x2d*255.0).round(),0,255).astype(np.uint8)img=Image.fromarray(x255,mode="L")buf=io.BytesIO()img.save(buf,format="PNG",optimize=True)returnbase64.b64encode(buf.getvalue()).decode("ascii")defcompute_distances(a:np.ndarray,b:np.ndarray)->dict:"""Compute L1, L2, Linf, and elastic-net distances.

    Args:
        a, b: Arrays to compare

    Returns:
        dict: Dictionary containing l1, l2, linf, and elastic distance values
    """diff=a-b
    l1=float(np.sum(np.abs(diff)))l2=float(np.sqrt(np.sum(diff**2)))linf=float(np.max(np.abs(diff)))return{"l1":l1,"l2":l2,"linf":linf}# Fetch challenge and check clean predictionch=requests.get(f"{BASE_URL}/challenge",timeout=10).json()x=x01_from_b64_png(ch["image_b64"])# (28, 28)lab=int(ch["label"])# baseline labelbeta=float(ch["beta"])# elastic-net trade-offelastic_max=float(ch["elastic_max"])# elastic-net boundl2_max=float(ch["l2_max"])# L2 auxiliary boundl1_max=float(ch["l1_max"])# L1 auxiliary boundres=requests.post(f"{BASE_URL}/predict",json={"image_b64":b64_png_from_x01(x)},timeout=10).json()print({"baseline_label":lab,"server_pred":res["pred"],"beta":beta,"elastic_max":elastic_max,"l2_max":l2_max,"l1_max":l1_max})`
```

### Model Architecture and Loading

The server uses this architecture. You need to replicate it locally to compute gradients and run FISTA.

Code:python```
`classSimpleClassifier(nn.Module):def__init__(self,num_classes:int=10):super().__init__()self.conv1=nn.Conv2d(1,32,3,1)self.conv2=nn.Conv2d(32,64,3,1)self.dropout1=nn.Dropout(0.25)self.dropout2=nn.Dropout(0.5)self.fc1=nn.Linear(9216,128)self.fc2=nn.Linear(128,num_classes)defforward(self,x:torch.Tensor)->torch.Tensor:"""Forward pass without normalization.

        Args:
            x: Input tensor (already normalized) with shape (N, 1, 28, 28)

        Returns:
            Logits with shape (N, 10)
        """x=torch.relu(self.conv1(x))x=torch.relu(self.conv2(x))x=torch.max_pool2d(x,2)x=self.dropout1(x)x=torch.flatten(x,1)x=torch.relu(self.fc1(x))x=self.dropout2(x)x=self.fc2(x)returnxdefmnist_normalize(x01:torch.Tensor)->torch.Tensor:"""Normalize [0,1] tensor to MNIST statistics.

    Args:
        x01: Input tensor in [0,1] range with shape (N, 1, 28, 28)

    Returns:
        Normalized tensor
    """ifx01.ndim!=4orx01.shape[1]!=1:raiseValueError(f"Expected (N,1,28,28), got{x01.shape}")ifx01.min()<0.0orx01.max()>1.0:raiseValueError("Input must be in [0,1]")return(x01-MNIST_MEAN)/MNIST_STD# Download and load weightswt=requests.get(f"{BASE_URL}/weights",timeout=10).contentopen("elasticnet_weights.pth","wb").write(wt)model=SimpleClassifier().eval()state=torch.load("elasticnet_weights.pth",map_location=torch.device("cpu"))model.load_state_dict(state)# Verify model works locallyx_tensor=torch.from_numpy(x[None,None,...]).float()logits=model(mnist_normalize(x_tensor))local_pred=int(torch.argmax(logits,dim=1).item())print(f"Local prediction:{local_pred}, should match server:{res['pred']}")`
```

### Testing Server Validation

Verify server-side checks by submitting the clean image and observing expected failures.

Code:python```
`bad=requests.post(f"{BASE_URL}/submit",json={"image_b64":b64_png_from_x01(x)},timeout=10)print(bad.status_code,bad.text)# expected: 400 with "Not misclassified"`
```
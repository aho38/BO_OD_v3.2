## <div align='left'>Introduction</div>

Using bayesian optimization to generate adversarial examples that targets Object Detectors.

## <div align='left'>Get Started</div>

<details open>
  <summary>Install</summary>
  Use <strong>Python>=3.7.0</strong> for your conda environment or <code>conda create --name [env_name] python=3.7</code>. Then clone the repository and install the required packages as the following:
  
  ```bash
  $ git clone git@github.com:aho38/LLNL-BO-lightprojection-attack.git
  $ cd LLNL-BO-lightprojection-attack
  $ pip install -r requirements.txt
  ```
  
</details>


<details open>
  <summary> Bayesian Optimization</summary>
  <a href='https://botorch.org/'>Botorch</a> is the main bayesian optimization package used in the code. You can find the specifications in <a href='https://github.com/aho38/BO_OD_v3.2/blob/master/utils/bo_utils.py'>bo_utils.py</a>. In this version of the code. Only <a href='https://github.com/aho38/BO_OD_v3.2/blob/master/utils/bo_utils.py#L62'>get_fitted_model</a> is used but the steps are the same:
<ol>
  
 </ol>
  
</details>

## <div align='left'>Adversarial Attack</summary>

**Digital Attack with Perlin Noise**
  
The file the runs digital attack with Perlin Noise on <a href='https://cocodataset.org/#home'>COCO Dataset</a>

```bash
$ python main.py
```

<details close>
  <summary>Digital Attack parameters </summary>
  
  There are numbers of parameters can be adjusted.
  
  ``` python
  def parse_opt():
    parser = argparse.ArgumentParser(prog='val.py')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_false', help='save a COCO-JSON results file')
    parser.add_argument('--project', default='runs/testing', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--att-type', type=str, default='mislabel-ml', help='types of attack')
    parser.add_argument('--start-count', type=int, default=0, help='number of start count usually 0-10')
    parser.add_argument('--num-count', type=int, default=5000, help='number of images that will be processed')
    parser.add_argument('--query-budget', type=int, default=20, help='number of queries we use')
    parser.add_argument('--norm', type=int, default=16, help='max norm of the image being perturbed')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    opt.data = check_file(opt.data)  # check file
    return opt
  ```
  
</details>

## Object Detect

```bash
$ python detect.py --source 0
```

"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
from main import process_batch_loss, compute_loss
import sys
import time
from pathlib import Path
import numpy as np

import cv2
from screeninfo import get_monitors
import torch
import torchvision
from torch._C import dtype
import torch.backends.cudnn as cudnn

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync, gen_rand_tensor, init_torch_seeds
from utils.perlin import noise_generator
from utils.bo_utils import get_fitted_model, gen_initial_data
from botorch.optim import optimize_acqf
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler



def grab_frame(cap):
    ret,frame = cap.read()
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

def process_pred(pred, path, img, im0s, dataset, save_dir, names,
                save_crop=False, save_txt=False, save_conf=False, save_img=False, view_img = False, 
                hide_labels=False, hide_conf=False, line_thickness=3, webcam=False):
    # Process predictions
    for i, det in enumerate(pred):  # detections per image
        if webcam:  # batch_size >= 1
            p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
        else:
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
    return i, p, im0, save_path

def process_img(img, device, half, pt):
    if pt:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    return img     


# @torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source=0,  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        no_webcam=False, # whether to show webcam
        pause_time=1, # time pause between frame for noise to render
        full_screen=False,
        num_queries=100,
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric()

    # ======== Directories ========
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # ======== Initialize ========
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # ======== Load model ========
    w = weights[0] if isinstance(weights, list) else weights
    classify, pt = False, w.endswith('.pt')  # inference type
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # ======== Dataloader ========
    if webcam:
        view_img = check_imshow() if not no_webcam else False
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    

    # ======== Initialize seed ========
    seed = 0
    init_torch_seeds(seed=seed)

    # ======== Run inference ========
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    # ======== initialize img window ========
    screen = get_monitors()[0] # get monitor information
    window_name = 'window'
    win_x, win_y = [screen.width - 1, screen.height - 1 ]
    if full_screen:
        cv2.namedWindow('window', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN)
    
    # =========== obtain initial solution ===========
    input("Press ENTER when setup is ready...")

    for frame_i, (path, img, im0s, vid_cap) in enumerate(dataset):
        # preprocess frames from image
        img = process_img(img, device, half, pt)
        

        # Inference
        with torch.no_grad():
            if pt:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(img, augment=augment, visualize=visualize)[0]

        # NMS
        pred, _ = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        pred[:] = [torch.cat((i[0:,0:4],i[0:,5:]), 1) for i in pred]

        _, p, im0, _ = process_pred(pred, path, img, im0s, dataset, save_dir, names, save_crop=save_crop, save_txt=save_txt, 
                    save_conf=save_conf, save_img=save_img, view_img=view_img, hide_labels=hide_labels, hide_conf=hide_conf, 
                    line_thickness=line_thickness, webcam=webcam)
        
        # Stream results
        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        gt_setup = input("Is the ground truth ready for training? [T/F]: \n") # ground truth setup. 

        if gt_setup == 'T':
            labels = torch.cat((pred[0][0:, -1:], pred[0][0:,0:4]), 1) # target labels
            break
        elif gt_setup == 'F':
            continue
        else:
            print("Invalid Input \n")
            continue

    # =========== Training starts ===========
    input("Press ENTER to let the training begin...")

    for frame_i, (path, img, im0s, vid_cap) in enumerate(dataset):

        # preprocess frames from image
        img = process_img(img, device, half, pt)

        ## ====================== BO TRAINING ======================
        ### Initial Query
        if frame_i < 5:
            ## initialize train param
            if frame_i == 0:
                train_param = torch.zeros((5, 4)).to(img) 
            ## period_x, period_y, and freq are all positive but smaller than d'. octaves are [1,4]
            train_param = gen_initial_data(frame_i, train_param, img)
            ## whether full screen noise or a certain size of noise
            generated_noise = noise_generator(*img.size()[-2:], period_x=train_param[frame_i,0], period_y=train_param[frame_i,1], octave=train_param[frame_i,2], freq=train_param[frame_i,3])
            generated_noise = torchvision.transforms.functional.resize(generated_noise[None, None], size=(int(win_y * 2),int(win_x * 2)))

            
            generated_noise = np.array(generated_noise.squeeze())
            cv2.imshow('window', generated_noise)
        ### Query starts
        else:
            try: # to avoid singular covariant matrix in GP
                gp_model = get_fitted_model(train_x=train_param, train_obj=train_obj, state_dict=state_dict)
            except:
                print("ERROR IN GP HAS OCCURED: please ensure covariant matrix is positive definite")
                pass
            qmc_sampler = SobolQMCNormalSampler(num_samples=200, seed=seed)
            qEI = qExpectedImprovement(gp_model, best_f=best_obj, sampler=qmc_sampler)
            candidates, _ = optimize_acqf(
                                acq_function=qEI,
                                bounds=torch.stack([
                                    torch.tensor([20.0, 20.0, 0.51, 4.0]).to(img),
                                    torch.tensor([160.0, 160.0, 4.5, 32.0]).to(img),
                                ]),
                                q=1,
                                num_restarts=10,
                                raw_samples=100,
                                )

            for i in range(len(candidates)):


                generated_noise = noise_generator(*img.shape[-2:], period_x=candidates[i,0], period_y=candidates[i,1], octave=candidates[i,2], freq=candidates[i,3])
                generated_noise = torchvision.transforms.functional.resize(generated_noise[None, None], size=(int(win_y * 2),int(win_x * 2)))

            
            generated_noise = np.array(generated_noise.squeeze())
            cv2.imshow('window', np.array(generated_noise))

            # concat training param used to generate noise
            train_param = torch.cat((train_param, candidates.to(img)),0)
        # save state_dict if GP models starts generating results
        state_dict = None if frame_i < 5 else gp_model.state_dict()


        # ==== pause time ====
        cv2.waitKey(100) # pause for the frame to rander 
        # =========== Obtain Frame with updated noise ============
        (path, img, im0s, vid_cap) = dataset.__next__()

        # ============== Process updated frames ==============
        img = process_img(img, device, half, pt)

        # ==================== Inference ====================
        t1 = time_sync()
        with torch.no_grad():
            if pt:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(img, augment=augment, visualize=visualize)[0]
            else: 
                print('check model file type... not .pt')
        
        ## ====================== Loss computation ======================
        loss_pred, _ = non_max_suppression(pred, 0.001, 0.6, classes, agnostic_nms, max_det=max_det)

        nl = len(labels)
        predn = torch.cat((loss_pred[0][:, 0:4], loss_pred[0][:,5:]),1)

        mask, correct_labels = process_batch_loss(predn, labels, torch.tensor(0.75).to(img))

        loss = compute_loss(torch.cat((loss_pred[0][:,4:5], predn[:,4:]), 1), mask, 'fabricate')

        if frame_i == 0:
            train_obj = loss[None,None]
        else:
            train_obj = torch.cat((train_obj, loss[None, None]), 0)

        if frame_i >= 4:
            best_obj, indx = train_obj.max(0)
            best_pred = train_param[indx]
        
        # ================================================================

        # NMS
        pred, _ = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        print(pred[0].size())
        pred[:] = [torch.cat((i[0:,0:4],i[0:,5:]), 1) for i in pred]
        t2 = time_sync()

        for i, det in enumerate(pred):  # detections per image
            print(i)
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        if frame_i == (num_queries - 1):
            break

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')
            


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--no-webcam',action='store_true', help='show webcam')
    parser.add_argument('--pause-time', type=int, default=0, help='time paused for noise to render')
    parser.add_argument('--full-screen', action='store_true', help='whether to display image fullscreen ')
    parser.add_argument('--num_queries', type=int, default=100, help='number of queries before breaking out of the loop')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def intersected(xyxy1, xyxy2):
    put_inside = False
    inter_area = 0
    
    # xyxy1 sebagai object structure, xyxy2 sebagai object damage
    # case jika di dalam
    if(xyxy1[0] <= xyxy2[0] and xyxy1[1] <= xyxy2[1] and xyxy1[2] >= xyxy2[2] and xyxy1[3] >= xyxy2[3]):
        put_inside = True
    
    # note
    # 0 => x min
    # 1 => y min
    # 2 => x max
    # 3 => y max
    # case jika berisisan (x2 y2 min)
    if(xyxy1[0] <= xyxy2[0] and xyxy2[0] <= xyxy1[2] and xyxy1[1] <= xyxy2[1] and xyxy2[1] <= xyxy1[3]):
        put_inside = True
    # case jika beririsan (x2 y2 max)
    if(xyxy1[0] <= xyxy2[2] and xyxy2[2] <= xyxy1[2] and xyxy1[1] <= xyxy2[3] and xyxy2[3] <= xyxy1[3]):
        put_inside = True
    # case jika beririsan (x2 min y2 max)
    if(xyxy1[0] <= xyxy2[0] and xyxy2[0] <= xyxy1[2] and xyxy1[1] <= xyxy2[3] and xyxy2[3] <= xyxy1[3]):
        put_inside = True
    # case jika beririsan (x2 max y2 min)
    if(xyxy1[0] <= xyxy2[2] and xyxy2[2] <= xyxy1[2] and xyxy1[1] <= xyxy2[1] and xyxy2[1] <= xyxy1[3]):
        put_inside = True
        
    if(put_inside):
        xA = int(max(xyxy1[0], xyxy2[0]))
        yA = int(max(xyxy1[1], xyxy2[1]))
        xB = int(min(xyxy1[2], xyxy2[2]))
        yB = int(min(xyxy1[3], xyxy2[3]))
        
        # compute intersected area
        inter_area = abs(xB - xA) * abs(yB - yA)
    return put_inside, inter_area

def structure_condition(structures):
    print('we are in structures')
    key = 0
    for obj in structures:
        key += 1 
        print(obj['co'])
        new_co = []
        for co in obj['co']:
            new_co.append(int(co))
        print(new_co)
        # remove co
        obj['co'] = f'struktur{key}'
        obj['co'] = new_co
        # translating coordinates
        
        print('damage :', len(obj['inside']))
        if len(obj['inside']) > 0:
            crack_percent = 0
            spalling_percent = 0
            for dmg in obj['inside']:
                dmg_percent = dmg['size']/obj['size']*100
                if dmg['name'] == 'crack':
                    crack_percent += dmg_percent
                if dmg['name'] == 'spalling':
                    spalling_percent += dmg_percent
            if crack_percent > 0:
                obj['damage'].append({'name':'crack', 'size':crack_percent})
            if spalling_percent > 0:
                obj['damage'].append({'name':'spalling', 'size':spalling_percent})
            
            if(crack_percent == 0 and spalling_percent == 0):
                obj['condition'] = 'Baik'
            elif(crack_percent < 25 or spalling_percent < 25):
                obj['condition'] = 'Rusak Ringan'
            elif(crack_percent < 50 or spalling_percent < 50):
                obj['condition'] = 'Rusak Sedang'
            else:
                obj['condition'] = 'Rusak Berat'
            
            # obj['condition'] = 'Rusak Berat'
                
    return structures
            

# def detect_size(save_img=False):
def detect_size(
        weights=None,
        source='inference/images',
        img_size=640,
        conf_thres=0.25,
        iou_thres=0.45,
        device='',
        view_img=False,
        save_txt=False,
        save_conf=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        update=False,
        project='runs/detect',
        name='result',
        exist_ok=False,
        no_trace=False
    ):
    # source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    imgsz, trace = img_size, not no_trace
    # save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    # device = select_device(opt.device)
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        # model = TracedModel(model, device, opt.img_size)
        model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # persiapan untuk menambahkan object dan struktur
    pillars_co = []
    objects_co = {}
    for name in names:
        objects_co[name] = {'name':name, 'co':[]}
        # objects_co.append({'name':name, 'co':[]})
    
    # classes_to_filter = ['pillar']
    # opt_classes = None
    # if classes_to_filter:
    #     opt_classes = []
    #     for class_name in classes_to_filter:
    #         opt_classes.append(names.index(class_name))
            
    # print(10*'-')
    # print('check out the names')
    # print(names)
    # print(10*'-')
    # print('check out the colors')
    # print(colors)
    # print(10*'-')
    # print('check out the opt_classes')
    # print(opt_classes)
    # print(10*'-')
    # print('check out the objects_co')
    # print(objects_co)
    # print(10*'-')
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                # model(img, augment=opt.augment)[0]
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        # pred = model(img, augment=opt.augment)[0]
        pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt_classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # memasukkan semua hasil deteksi kedalam koordinat
                    objects_co[names[int(cls)]]['co'].append(xyxy)
                    
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        
    # kez, memasukkan coordinat pillar
    for obj in objects_co['pillar']['co']:
        box_size = abs(int(obj[2]) - int(obj[0])) * abs(int(obj[3]) - int(obj[1]))
        pillars_co.append( {'co':obj, 'size': box_size, 'inside':[], 'condition':'Baik', 'damage':[]} )
        
    # kez, memasukkan object lain ke dalam pillar
    for obj in objects_co['crack']['co']:
        for pillco in pillars_co:
            # comparing if obj inside pillco
            is_inter, size = intersected(pillco['co'], obj)
            if(is_inter):
                pillco['inside'].append({'name':'crack', 'size':size})
                
    # kez, memasukkan object lain ke dalam pillar
    for obj in objects_co['spalling']['co']:
        for pillco in pillars_co:
            # comparing if obj inside pillco
            is_inter, size = intersected(pillco['co'], obj)
            if(is_inter):
                pillco['inside'].append({'name':'spalling', 'size':size})
    
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    
    pillars_co = structure_condition(pillars_co)
    return pillars_co


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect_size()
                strip_optimizer(opt.weights)
        else:
            detect_size()

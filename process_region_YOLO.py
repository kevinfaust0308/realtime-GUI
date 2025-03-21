import mss
import numpy as np
import cv2

from utils import extract_tiles

def process_region(region, **kwargs):

    metadata = kwargs['metadata']
    model = kwargs['model']

    ###

    tile_size = metadata['tile_size']

    with mss.mss() as sct:
        screenshot = sct.grab(region)

    frame = np.array(screenshot, dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    frame = frame[:max((frame.shape[0]//tile_size)*tile_size, tile_size), :max((frame.shape[1]//tile_size)*tile_size, tile_size), :]

    slices = extract_tiles(frame, tile_size)

    # Detect and render
    results = model(slices)

    from ultralytics.utils.plotting import Annotator
    from ultralytics.data.augment import LetterBox
    import torch



    tile_size_y = tile_size_x = tile_size
    if frame.shape[0] < tile_size:
        tile_size_y = frame.shape[0]
    if frame.shape[1] < tile_size:
        tile_size_x = frame.shape[1]

    seg_mask = np.zeros(frame.shape)
    k = 0
    for i in range(frame.shape[0] // tile_size_y):
        for j in range(frame.shape[1] // tile_size_x):
            # need to override so that i can choose mask colors...
            if results[k].masks is not None and results[k].masks.shape[0] != 0:
                annotator = Annotator(
                    np.ascontiguousarray(results[k].orig_img),
                    line_width=None,
                    font_size=None,
                    font="Arial.ttf",
                    pil=False,
                    example=results[0].names,
                )
                img = LetterBox(results[k].masks.shape[1:])(image=annotator.result())
                im_gpu = (
                        torch.as_tensor(img, dtype=torch.float16, device=results[k].masks.data.device)
                        .permute(2, 0, 1)
                        .flip(0)
                        .contiguous()
                        / 255
                )
                colors = {
                    0: (0, 0, 255),  # positive. red
                    1: (255, 0, 0),  # negative. blue
                    2: (0, 255, 0),  # misc. green
                }
                annotator.masks(results[k].masks.data,
                                colors=[colors[x] for x in results[k].boxes.cls.cpu().numpy()], im_gpu=im_gpu)

                seg_mask[
                (i * tile_size):((i * tile_size) + tile_size),
                (j * tile_size):((j * tile_size) + tile_size),
                :
                # ] = results[k].plot(labels=False, boxes=False)
                ] = annotator.result()

            else:
                # no masks found
                seg_mask[
                (i * tile_size):((i * tile_size) + tile_size),
                (j * tile_size):((j * tile_size) + tile_size),
                :
                ] = frame[
                    (i * tile_size):((i * tile_size) + tile_size),
                    (j * tile_size):((j * tile_size) + tile_size),
                    :
                    ]

            k += 1

    num_pos = 0
    num_pos_neg = 0
    for r in results:
        num_pos_curr = torch.sum(r.boxes.cls == 0).cpu().numpy()
        num_pos += num_pos_curr
        num_pos_neg += num_pos_curr + torch.sum(r.boxes.cls == 1).cpu().numpy()

    text = '(+) {:.2f} %\n'.format(num_pos / num_pos_neg * 100 if num_pos_neg > 0 else 0)
    text += '(+) cells: {}\n'.format(num_pos)
    text += '(-) cells: {}\n'.format(num_pos_neg - num_pos)

    return seg_mask.astype(np.uint8), text

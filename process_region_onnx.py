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
    frame = frame[:max((frame.shape[0] // tile_size) * tile_size, tile_size),
            :max((frame.shape[1] // tile_size) * tile_size, tile_size), :]

    slices = extract_tiles(frame, tile_size)

    batch = np.divide(np.array(slices), 255)

    inputs = {model.get_inputs()[0].name: batch.astype(np.float32)}
    confs = model.run(None, inputs)[0]

    confs = np.mean(confs, axis=0)
    top_3_idx = np.argsort(-confs)[:3]

    res = ''
    for idx in top_3_idx:
        if confs[idx] < float(kwargs['additional_configs']['min_conf']):
            break

        res += '{}: {:.4f}\n'.format(metadata['classes'][idx], confs[idx])
    print(res)

    return frame, res
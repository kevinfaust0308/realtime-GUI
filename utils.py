def extract_tiles(frame, tile_size):
    slices = []

    # capture_size > tile_size: split capture window into sub tiles
    # capture_size < tile_size: no splitting. use the capture window dimension.
    # - there is a message on gui telling user that choosing a capture size less than the trained tile size is not recommended...

    tile_size_y = tile_size_x = tile_size

    if frame.shape[0] < tile_size:
        tile_size_y = frame.shape[0]
    if frame.shape[1] < tile_size:
        tile_size_x = frame.shape[1]

    for i in range(frame.shape[0] // tile_size_y):
        for j in range(frame.shape[1] // tile_size_x):
            tile_slice = frame[
                         (i * tile_size_y):((i * tile_size_y) + tile_size_y),
                         (j * tile_size_x):((j * tile_size_x) + tile_size_x),
                         :
                         ]
            slices.append(tile_slice)

    return slices


def load_model(model_info):

    if model_info['repo_src'] == 'HuggingFace':
        if model_info['model'] == 'VGG19':
            from huggingface_hub import from_pretrained_keras

            model = from_pretrained_keras(model_info['repo'])

            from process_region_keras import process_region
            return model, process_region

    elif model_info['repo_src'] == 'Local':
        if model_info['model'] == 'YOLO':
            from ultralytics import YOLO

            model = YOLO(model_info['repo'])

            from process_region_YOLO import process_region
            return model, process_region


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

    res = {
        'model': None,
        'process_region_func': None,
        'using_gpu': False
    }

    if model_info['repo_src'] == 'HuggingFace':
        if model_info['model'] == 'ONNX':

            from huggingface_hub import hf_hub_download
            # Import torch will preload necessary DLLs. It needs to be done before creating session.
            # REQUIRED FOR GPU TO WORK
            import torch
            import onnxruntime as ort

            model_path = hf_hub_download(repo_id=model_info['repo'], filename="model.onnx")

            # Load ONNX model with GPU support if available
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            session = ort.InferenceSession(model_path, providers=providers)

            from process_region_onnx import process_region

            res['model'] = session
            res['process_region_func'] = process_region
            res['using_gpu'] = 'CUDAExecutionProvider' in available_providers

            return res

    elif model_info['repo_src'] == 'Local':
        if model_info['model'] == 'YOLO':
            from ultralytics import YOLO
            import torch
            from process_region_YOLO import process_region

            res['model'] = YOLO(model_info['repo'])
            res['process_region_func'] = process_region
            res['using_gpu'] = torch.cuda.is_available()

            return res

    return res

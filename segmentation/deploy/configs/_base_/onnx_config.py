onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {
            0: 'batch'
        },
        'output': {
            0: 'batch'
        }
    },
    input_shape=None,
    optimize=False)

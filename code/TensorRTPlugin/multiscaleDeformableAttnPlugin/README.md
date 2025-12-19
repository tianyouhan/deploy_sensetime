# multiscaleDeformableAttn Plugin

**Table Of Contents**
- [Compile](#compile)
- [Use Plugin Lib](#use-plugin-lib)
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Compile
```bash
mkdir build
cd build
cmake ..
make
```

编译后会生成libmultiscaleDeformableAttnPlugin.so

## Use Plugin Lib

将so放到senseauto-perception-camera/node/resource/models/perception_camera/thor/trt_plugin/目录下，node在编译打包后会自动安装到板子上/opt/senseauto_active/senseauto-perception-camera/resource/models/thor/trt_plugin/目录下

在放置好so后需要配置prototxt在运行时调用。这里trt_plugin_infos是需要调用的插件so的信息：

- library_path：代表插件so在板子上路径
- plugin_name；代表插件的名字

这里以libmultiscaleDeformableAttnPlugin.so示例：

```protobuf
###### spetr backbone
operators{
  name: "SPETRPredictorOP_Backbone"
  type: "NVGeneralPredictorOP"
  top: "GeneralInputSetupOP"
  general_predictor_param {
    gpu_id: 0
    dnn_param {
      model_folder_path: "models/thor/vd_dlp/perception_0826_cityway_highway"
      model_file: "backbone.bin"
      model_weight: ""
      model_config: "parameters_backbone.json"
    }
    net_input_name: [
      "center_camera_fov30",
      "center_camera_fov120",
      "left_front_camera",
      "left_rear_camera",
      "rear_camera",
      "right_rear_camera",
      "right_front_camera"
    ]
    data_input_name: [
        "center_camera_fov30_permuted_1",
        "center_camera_fov120_permuted_1",
        "left_front_camera_permuted_1",
        "left_rear_camera_permuted_1",
        "rear_camera_permuted_1",
        "right_rear_camera_permuted_1",
        "right_front_camera_permuted_1"
    ]
    src_image_type: FLOAT32
    predictor_type: TRT_CUDA_GRAPH
    trt_plugin_infos {
      library_path: "/opt/senseauto_active/senseauto-perception-camera/resource/models/thor/trt_plugin/libmultiscaleDeformableAttnPlugin.so"
      plugin_name: "MultiscaleDeformableAttnPlugin_TRT"
    }
  }
  execute_priority: -2
}
```





## Description

The `multiscaleDeformableAttnPlugin` is used to perform attention computation over a small set of key sampling points around a reference point rather than looking over all possible spatial locations. It makes use of multiscale feature maps to effectively represent objects at different scales. It helps to achieve faster convergence and better performance on small objects. 

### Structure

The `multiscaleDeformableAttnPlugin` takes 5 inputs in the following order :  `value`, `spatial_shapes`, `level_start_index`, `sampling_locations`, and `atttention_weights`.

`value` 
The input feature maps from different scales concatenated to provide the input feature vector. The shape of this tensor is `[N, S, M, D]` where `N` is batch size, `S` is the length of the feature maps, `M` is the number of attentions heads, `D` is hidden_dim/num_heads.

`spatial_shapes`
The shape of each feature map. The shape of this tensor is `[L, 2]` where `L` is the number of feature maps.

`level_start_index`
This tensor is used to find the sampling locations for different feature levels as the input feature tensors are flattened. The shape of this tensor is `[L,]`.

`sampling_locations`
This tensor acts as a pre-filter for prominent key elements out of all the feature map pixels. The shape of this tensor is `[N, Lq, M, L, P, 2]` where `P` is the number of points, `Lq` is the length of feature maps(encoder)/length of queries(decoder).

`attention_weights`
This tensor consists of the attention weights whose shape is `[N, Lq, M, L, P]`.

The `multiscaleDeformableAttnPlugin` generates the attention output of shape `[N, Lq, M, D]`.

## Parameters

`multiscaleDeformableAttnPlugin` has plugin creator class `multiscaleDeformableAttnPluginCreator` and plugin class `multiscaleDeformableAttnPlugin`.

The plugin does not require any parameters to be built and used.


## Additional resources

The following resources provide a deeper understanding of the `multiscaleDeformableAttnPlugin` plugin:

**Networks:**
- [Deformable DETR](https://arxiv.org/pdf/2010.04159.pdf)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.

## Changelog 

Feb 2022 
This is the first release of this `README.md` file.

## Known issues 

There are no known issues in this plugin.

import os
import torch
from cutie.utils.get_default_model import get_default_model

class EncodeImage(torch.nn.Module):
    def __init__(self, cutie):
        super(EncodeImage, self).__init__()
        self.cutie = cutie

    def forward(self, x):
        #ms_image_feat = self.cutie.pixel_encoder(x)

        ms_features, pix_feat = self.cutie.encode_image(x)
        key, shrinkage, selection = self.cutie.transform_key(ms_features[0])

        return ms_features[0], ms_features[1], ms_features[2], pix_feat, key, shrinkage, selection

class EncodeMask(torch.nn.Module):
    def __init__(self, cutie):
        super(EncodeMask, self).__init__()
        self.cutie = cutie

    def forward(self, image, pix_feat, sensory, masks, others):
        mask_value, new_sensory = self.cutie.mask_encoder(image, pix_feat, sensory, masks, others)
        return mask_value, new_sensory

class CutieOnnxExporter():
    def __init__(self, onnx_output_dir: str):
        self.onnx_output_dir = onnx_output_dir
        if not os.path.exists(self.onnx_output_dir):
            os.makedirs(self.onnx_output_dir, exist_ok=True)

        self.cutie = get_default_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cutie.to(self.device)
        self.cutie.eval()

    def export_to_onnx(self):
        try:
            print("Starting ONNX export...")
            # 创建示例输入
            image = torch.randn(1, 3, 480, 480).to(self.device)
            masks = torch.randn(1, 1, 480, 480).to(self.device)
            others = torch.randn(1, 1, 480, 480).to(self.device)
            sensory = torch.randn(1, 1, 256, 30, 30).to(self.device)  # 根据模型配置调整维度
            
            self.export_encoder_image(image)
            #self.export_mask_encoder(image, masks, others, sensory)
            #self.export_object_transformer(image)
            print("ONNX export completed successfully!")
        except Exception as e:
            print(f"Error during export: {str(e)}")

    def export_encoder_image(self, image):
        print("Starting export pixel encoder...")
        onnx_path = os.path.join(self.onnx_output_dir, 'encoder_image.onnx')
        encode_image = EncodeImage(self.cutie)

        torch.onnx.export(
            encode_image,
            image,
            onnx_path,
            verbose=True,
            input_names=['input'],
            output_names=['output_ms_features_f16', 'output_ms_features_f8', 'output_ms_features_f4', 'output_pix_feat', 'output_key', 'output_shrinkage', 'output_selection'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output_ms_features_f16': {0: 'batch_size'},
                'output_ms_features_f8': {0: 'batch_size'},
                'output_ms_features_f4': {0: 'batch_size'},
                'output_pix_feat': {0: 'batch_size'},
                'output_key': {0: 'batch_size'},
                'output_shrinkage': {0: 'batch_size'},
                'output_selection': {0: 'batch_size'}
            },
            opset_version=11,
            do_constant_folding=False
        )

    def export_encoder_mask(self, image, masks, others, sensory):
        onnx_path = os.path.join(self.onnx_output_dir, 'mask_encoder.onnx')
        mask_encoder = self.cutie.mask_encoder
        pix_feat = self.cutie.pixel_encoder(image)[0]  # 使用第一个特征图
        
        torch.onnx.export(
            mask_encoder,
            (image, pix_feat, sensory, masks, others),
            onnx_path,
            verbose=True,
            input_names=['image', 'pix_feat', 'sensory', 'masks', 'others'],
            output_names=['output_features', 'output_sensory'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'pix_feat': {0: 'batch_size'},
                'sensory': {0: 'batch_size'},
                'masks': {0: 'batch_size'},
                'others': {0: 'batch_size'},
                'output_features': {0: 'batch_size'},
                'output_sensory': {0: 'batch_size'}
            },
            opset_version=11,
            do_constant_folding=True
        )

    def export_object_transformer(self, pixel_readout):
        onnx_path = os.path.join(self.onnx_output_dir, 'object_transformer.onnx')
        obj_transformer = self.cutie.object_transformer
        
        # 创建示例输入
        batch_size = 1
        num_objects = 1
        embed_dim = self.cutie.cfg.model.value_dim
        h, w = 30, 30  # 根据模型配置调整
        
        pixel_readout = torch.randn(batch_size, num_objects, embed_dim, h, w).to(self.device)
        obj_memory = torch.randn(batch_size, num_objects, 8, embed_dim, h, w).to(self.device)  # 8是时序长度
        
        torch.onnx.export(
            obj_transformer,
            (pixel_readout, obj_memory),
            onnx_path,
            verbose=True,
            input_names=['pixel_readout', 'obj_memory'],
            output_names=['output', 'aux_outputs'],
            dynamic_axes={
                'pixel_readout': {0: 'batch_size', 1: 'num_objects'},
                'obj_memory': {0: 'batch_size', 1: 'num_objects', 2: 'time_steps'},
                'output': {0: 'batch_size', 1: 'num_objects'},
                'aux_outputs': {0: 'batch_size', 1: 'num_objects'}
            },
            opset_version=11,
            do_constant_folding=True
        )

if __name__ == '__main__':
    onnx_output_dir = 'onnx_export'
    cutie_onnx_exporter = CutieOnnxExporter(onnx_output_dir)
    cutie_onnx_exporter.export_to_onnx()

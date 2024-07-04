from classification import Classification 

onnx_save_path  = "model_data/models.onnx"
opset           = 12
simplify        = False
if __name__ == "__main__":
    classification = Classification(model_path  = 'logs/loss_2024_06_29_09_46_40/best_epoch_weights.pth',
                                    input_shape = [48, 1200],
                                    backbone    = 'resnet18')
    print(classification.backbone)
    classification.convert_to_onnx(onnx_save_path, opset, simplify)
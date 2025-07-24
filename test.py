from rapidocr import RapidOCR

# 步骤2中的1.yaml
config_path = "./default_rapidocr.yaml"
engine = RapidOCR(config_path=config_path)

# img_url = "output_images/slice_000.jpg"
# img_url = "debug_images/slice_chengcuqiepian.jpg"
img_url = "debug_images/slice_chengxuneibaocun.jpg"
result = engine(img_url)
print(result)
result.vis("debug_images_output/chengxuneibaocun_ocr_test.jpg")
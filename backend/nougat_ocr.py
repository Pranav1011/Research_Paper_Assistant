import torch
from transformers import AutoProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
from PIL import Image

processor = AutoProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base").eval()

def ocr_with_nougat(pdf_path: str) -> str:
    images = convert_from_path(pdf_path)
    result = []

    for image in images:
        image = image.convert("RGB")
        pixel_values = processor(images=[image], return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result.append(decoded.strip())

    return "\n\n".join(result)
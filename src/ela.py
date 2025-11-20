from PIL import Image, ImageChops, ImageEnhance

def ela_image(path, quality=90):
    img = Image.open(path).convert("RGB")
    tmp = "/tmp/ela_tmp.jpg"
    img.save(tmp, "JPEG", quality=quality)
    resaved = Image.open(tmp)
    ela = ImageChops.difference(img, resaved)
    max_diff = max([ex[1] for ex in ela.getextrema()])
    scale = 255.0 / max(1, max_diff)
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    return ela

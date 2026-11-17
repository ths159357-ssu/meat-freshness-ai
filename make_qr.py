# 앱 주소로 QR 코드 만들기:  python make_qr.py https://아이디.github.io/meat-freshness-ai/
import sys, qrcode
from PIL import Image, ImageDraw, ImageFont
url = sys.argv[1]
qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=16, border=3)
qr.add_data(url); qr.make(fit=True)
img = qr.make_image(fill_color="#0f3d3e", back_color="white").convert("RGB")
W, H = img.size
out = Image.new("RGB", (W, H + 110), "white"); out.paste(img, (0, 0))
d = ImageDraw.Draw(out)
def font(names, size):
    for n in names:
        for base in ["/usr/share/fonts/opentype/noto/", "/usr/share/fonts/truetype/noto/", "/usr/share/fonts/truetype/nanum/", "C:/Windows/Fonts/", "/System/Library/Fonts/"]:
            try: return ImageFont.truetype(base + n, size)
            except OSError: pass
    return None
f1 = font(["NotoSansCJK-Bold.ttc", "NanumGothicBold.ttf", "malgunbd.ttf", "AppleSDGothicNeo.ttc"], 40)
f2 = font(["NotoSansCJK-Regular.ttc", "NanumGothic.ttf", "malgun.ttf", "AppleSDGothicNeo.ttc"], 22) or ImageFont.load_default()
if f1: d.text((W / 2, H + 10), "육류 신선도 판단 앱", fill="#0f3d3e", font=f1, anchor="mt")   # 한글 글꼴이 있을 때만 제목 표시
d.text((W / 2, H + 64), url, fill="#6c7a89", font=f2, anchor="mt")
out.save("qr_app.png"); print("qr_app.png 저장:", url)

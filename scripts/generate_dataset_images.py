#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 datasets/ 全部测试图片 + assets/sample.jpg 示例图片。
图片用 PIL 合成，每张带真实可判别的视觉特征（版式、颜色、标题），不是纯文字猜测。
"""
from PIL import Image, ImageDraw, ImageFont

FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def blank(w=640, h=800, bg="white"):
    img = Image.new("RGB", (w, h), bg)
    return img, ImageDraw.Draw(img)


def ruled(d, box, gap=36, color="#b9c4d0"):
    """画横线（信纸/报表）"""
    x0, y0, x1, y1 = box
    y = y0
    while y < y1:
        d.line([(x0, y), (x1, y)], fill=color, width=1)
        y += gap


def grid(d, box, rows, cols, color="#9aa7b4"):
    """画表格网格（发票/报表）"""
    x0, y0, x1, y1 = box
    for i in range(rows + 1):
        y = y0 + (y1 - y0) * i // rows
        d.line([(x0, y), (x1, y)], fill=color, width=1)
    for j in range(cols + 1):
        x = x0 + (x1 - x0) * j // cols
        d.line([(x, y0), (x, y1)], fill=color, width=1)


# ── 文档分类 ─────────────────────────────────────────────────────────

def doc_invoice():
    img, d = blank(bg="#fdfdf8")
    t = font(FONT_BOLD, 40)
    r = font(FONT_REG, 22)
    d.text((180, 50), "TAX INVOICE", font=t, fill="#1a3a6b")
    d.line([(60, 110), (580, 110)], fill="#1a3a6b", width=4)
    d.text((60, 130), "Invoice No: INV-2026-0047", font=r, fill="black")
    d.text((60, 160), "Date: 2026-09-23", font=r, fill="black")
    grid(d, (60, 230, 580, 470), rows=5, cols=4)
    cols = ["Item", "Qty", "Price", "Total"]
    c = font(FONT_BOLD, 20)
    for j, name in enumerate(cols):
        d.text((75 + j * 130, 240), name, font=c, fill="black")
    rows = ["Widget A", "Widget B", "Widget C", "Widget D", "Widget E"]
    for i, name in enumerate(rows):
        d.text((75, 290 + i * 48), name, font=r, fill="black")
        d.text((205 + i * 0, 290 + i * 48), str(2 + i), font=r, fill="black")
        d.text((335, 290 + i * 48), f"${19 + i * 7}.00", font=r, fill="black")
        d.text((455, 290 + i * 48), f"${(2 + i) * (19 + i * 7)}.00", font=r, fill="black")
    d.text((60, 500), "TOTAL: $461.00", font=font(FONT_BOLD, 26), fill="#8b1a1a")
    d.text((60, 540), "Payment due within 30 days", font=r, fill="black")
    d.rectangle([50, 40, 590, 580], outline="#1a3a6b", width=2)
    return img


def doc_contract():
    img, d = blank(bg="#ffffff")
    t = font(FONT_BOLD, 36)
    r = font(FONT_REG, 20)
    d.text((140, 50), "SERVICE CONTRACT", font=t, fill="black")
    d.line([(140, 95), (500, 95)], fill="black", width=2)
    body = [
        "This Agreement is entered into by and between",
        "the Parties identified below. Party A hereby",
        "agrees to provide services to Party B under",
        "the terms and conditions set forth herein.",
        "",
        "1. TERM. The term of this Agreement shall",
        "   commence on the Effective Date and continue",
        "   for a period of twelve (12) months.",
        "2. PAYMENT. Party B shall pay Party A within",
        "   net thirty (30) days of invoice receipt.",
        "",
        "IN WITNESS WHEREOF, the Parties have executed",
        "this Agreement as of the date first written.",
    ]
    y = 130
    for line in body:
        d.text((70, y), line, font=r, fill="black")
        y += 30
    d.line([(100, 620), (280, 620)], fill="black", width=2)
    d.line([(360, 620), (540, 620)], fill="black", width=2)
    d.text((100, 628), "Party A Signature", font=r, fill="#555555")
    d.text((360, 628), "Party B Signature", font=r, fill="#555555")
    return img


def doc_report():
    img, d = blank(bg="#f4f6f9")
    t = font(FONT_BOLD, 34)
    r = font(FONT_REG, 20)
    d.text((150, 45), "QUARTERLY REPORT", font=t, fill="#0f4c5c")
    d.text((150, 90), "Q2 2026 Performance Summary", font=r, fill="#333333")
    # 柱状图
    bars = [(120, 180, 90), (220, 140, 90), (320, 120, 90), (420, 60, 90), (500, 100, 90)]
    for x, hgt, w in bars:
        d.rectangle([x, 380 - hgt, x + w, 380], fill="#0f4c5c")
    d.line([(100, 380), (600, 380)], fill="black", width=2)
    d.text((120, 395), "Jan  Feb  Mar  Apr  May", font=r, fill="#333333")
    # 折线图
    pts = [(120, 560), (220, 520), (320, 545), (420, 480), (520, 450)]
    d.line(pts, fill="#9a3c50", width=3)
    for p in pts:
        d.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill="#9a3c50")
    d.text((120, 600), "Key metrics trend, first half 2026", font=r, fill="#333333")
    d.text((120, 640), "Prepared by Analytics Team", font=r, fill="#666666")
    return img


def doc_letter():
    img, d = blank(bg="#fffcf2")
    r = font(FONT_REG, 22)
    d.text((80, 60), "April 5, 2026", font=r, fill="black")
    d.text((80, 120), "Dear Ms. Thompson,", font=r, fill="black")
    ruled(d, (80, 170, 560, 560), gap=40, color="#c8b88a")
    d.text((400, 600), "Sincerely,", font=r, fill="black")
    d.text((400, 660), "Daniel Hayes", font=font(FONT_BOLD, 22), fill="black")
    return img


def doc_receipt():
    img, d = blank(w=440, h=620, bg="#fffbf0")
    r = font(FONT_REG, 20)
    t = font(FONT_BOLD, 26)
    d.text((120, 30), "RECEIPT", font=t, fill="black")
    d.line([(40, 70), (400, 70)], fill="black", width=1)
    items = [("Coffee", "4.50"), ("Bagel", "3.25"), ("Juice", "5.00"), ("Tax", "1.28")]
    y = 90
    for name, price in items:
        d.text((50, y), name, font=r, fill="black")
        d.text((300, y), f"${price}", font=r, fill="black")
        y += 34
    d.line([(40, y), (400, y)], fill="black", width=1)
    d.text((50, y + 8), "TOTAL", font=t, fill="black")
    d.text((300, y + 8), "$14.03", font=t, fill="black")
    # 条形码
    x = 50
    import random
    random.seed(7)
    while x < 390:
        w = random.choice([2, 4, 6])
        d.rectangle([x, y + 90, x + w, y + 150], fill="black")
        x += w + random.choice([3, 5])
    d.text((130, y + 160), "Thank you!", font=r, fill="black")
    return img


# ── 场景识别 ─────────────────────────────────────────────────────────

def scene_indoor_1():
    img, d = blank(640, 420, "#e8e4da")
    # 木地板
    d.rectangle([0, 300, 640, 420], fill="#8f6b4a")
    for i in range(8):
        x = i * 80
        d.line([(x, 300), (x - 40, 420)], fill="#7a5a3d", width=2)
    # 地毯
    d.ellipse([200, 330, 440, 410], fill="#b0655c")
    # 沙发
    d.rectangle([140, 200, 500, 300], fill="#5a7d9a")
    d.rectangle([140, 180, 180, 300], fill="#4a6a86")
    d.rectangle([460, 180, 500, 300], fill="#4a6a86")
    # 挂画
    d.rectangle([280, 60, 360, 130], fill="#d9c8a9")
    d.rectangle([295, 75, 345, 115], fill="#9db4c0")
    # 台灯
    d.polygon([(80, 90), (130, 90), (105, 130)], fill="#e8d174")
    d.line([(105, 130), (105, 200)], fill="#666666", width=4)
    d.rectangle([85, 200, 125, 210], fill="#444444")
    return img


def scene_indoor_2():
    img, d = blank(640, 420, "#f2efe8")
    # 墙
    d.rectangle([0, 0, 640, 260], fill="#f2efe8")
    d.rectangle([0, 260, 640, 420], fill="#a08668")
    # 窗户
    d.rectangle([220, 70, 420, 210], fill="#cfe8ff", outline="#8a7a5c", width=6)
    d.line([(320, 70), (320, 210)], fill="#8a7a5c", width=4)
    d.line([(220, 140), (420, 140)], fill="#8a7a5c", width=4)
    # 桌椅
    d.rectangle([240, 280, 400, 295], fill="#6b4f36")
    d.rectangle([250, 295, 265, 390], fill="#5a4029")
    d.rectangle([375, 295, 390, 390], fill="#5a4029")
    # 书
    d.rectangle([300, 262, 360, 280], fill="#a03c3c")
    d.rectangle([290, 268, 300, 280], fill="#3c5aa0")
    return img


def scene_outdoor_1():
    img, d = blank(640, 420, "#a5d8f3")
    # 天空太阳
    d.ellipse([500, 40, 580, 120], fill="#f7d774")
    d.rectangle([0, 280, 640, 420], fill="#7cb56b")
    # 远山
    d.polygon([(0, 280), (160, 150), (330, 280)], fill="#6f8f7d")
    d.polygon([(250, 280), (460, 120), (640, 280)], fill="#5f7d6b")
    # 树
    for x in [60, 120, 560]:
        d.rectangle([x - 8, 220, x + 8, 290], fill="#6b4f36")
        d.ellipse([x - 55, 140, x + 55, 240], fill="#3f7d3a")
    # 小路
    d.polygon([(280, 420), (360, 420), (330, 280), (300, 280)], fill="#d9c8a0")
    return img


def scene_outdoor_2():
    img, d = blank(640, 420, "#b8e0f5")
    d.rectangle([0, 300, 640, 420], fill="#8fce8f")
    # 沙滩海
    d.rectangle([0, 200, 640, 300], fill="#4da3d9")
    d.rectangle([0, 300, 640, 340], fill="#e8d9a0")
    # 云
    for x, y in [(90, 60), (300, 90), (480, 50)]:
        d.ellipse([x, y, x + 90, y + 40], fill="white")
        d.ellipse([x + 40, y - 15, x + 130, y + 30], fill="white")
    # 棕榈树
    d.line([(110, 210), (90, 120)], fill="#6b4f36", width=8)
    for dx, dy in [(-60, -5), (-30, -30), (5, -40), (40, -25), (65, 0)]:
        d.ellipse([90 + dx, 120 + dy, 90 + dx + 55, 120 + dy + 18], fill="#2f8f4f")
    return img


def scene_night_1():
    img, d = blank(640, 420, "#0a1030")
    # 月亮
    d.ellipse([480, 50, 560, 130], fill="#f0e9c8")
    # 星星
    import random
    random.seed(42)
    for _ in range(40):
        x, y = random.randint(0, 630), random.randint(0, 240)
        d.point((x, y), fill="white")
        if random.random() < 0.2:
            d.point((x + 1, y), fill="#cccccc")
    # 楼房
    d.rectangle([80, 180, 200, 420], fill="#141c3c")
    d.rectangle([240, 120, 380, 420], fill="#101838")
    d.rectangle([420, 200, 540, 420], fill="#141c3c")
    # 窗户灯光
    for bx, by, bw, bh in [(80, 180, 120, 240), (240, 120, 140, 300), (420, 200, 120, 220)]:
        wy = by + 20
        while wy < by + bh - 20:
            for wx in range(bx + 15, bx + bw - 20, 30):
                import random as _r
                if _r.Random((wx * 31 + wy) & 0xFFFF).random() < 0.6:
                    d.rectangle([wx, wy, wx + 16, wy + 20], fill="#f5d76e")
            wy += 35
    # 路灯
    d.line([(600, 150), (600, 420)], fill="#333344", width=5)
    d.ellipse([588, 140, 612, 160], fill="#f5e05c")
    return img


# ── 内容审核 ─────────────────────────────────────────────────────────

def mod_normal_1():
    img, d = blank(640, 420, "white")
    t = font(FONT_BOLD, 30)
    r = font(FONT_REG, 22)
    d.text((180, 40), "Community Notice", font=t, fill="#2c5282")
    d.line([(80, 90), (560, 90)], fill="#2c5282", width=3)
    lines = [
        "The public library will host a free",
        "reading workshop every Saturday morning.",
        "All residents are welcome to join and",
        "bring their families. Refreshments will",
        "be served. No registration required.",
    ]
    y = 120
    for line in lines:
        d.text((80, y), line, font=r, fill="black")
        y += 40
    d.text((80, 360), "Contact: info@community.org", font=r, fill="#666666")
    return img


def mod_normal_2():
    img, d = blank(640, 420, "#f0f7f0")
    t = font(FONT_BOLD, 30)
    r = font(FONT_REG, 22)
    d.text((140, 40), "Fresh Vegetables Sale", font=t, fill="#2f7d32")
    d.line([(80, 90), (560, 90)], fill="#2f7d32", width=3)
    # 蔬菜
    d.ellipse([120, 140, 220, 240], fill="#e8702a")
    d.polygon([(260, 240), (300, 140), (340, 240)], fill="#7cb56b")
    d.ellipse([380, 140, 500, 240], fill="#d94f4f")
    d.rectangle([425, 240, 455, 280], fill="#4f7d3a")
    d.text((120, 300), "Tomatoes, lettuce, and peppers", font=r, fill="black")
    d.text((120, 340), "Farm-fresh, delivered daily", font=r, fill="black")
    return img


def mod_violation_1():
    img, d = blank(640, 420, "#2b2b2b")
    t = font(FONT_BOLD, 44)
    r = font(FONT_REG, 24)
    d.rectangle([0, 0, 640, 120], fill="#8b1a1a")
    d.text((140, 35), "BANNED SUBSTANCE", font=t, fill="white")
    lines = [
        "Street sale of illegal goods.",
        "Contact listed number for pickup.",
        "No questions asked. Cash only.",
    ]
    y = 160
    for line in lines:
        d.text((80, y), line, font=r, fill="#e0e0e0")
        y += 45
    d.text((80, 310), "Discretion advised.", font=r, fill="#c89090")
    return img


def mod_review_1():
    img, d = blank(640, 420, "white")
    t = font(FONT_BOLD, 30)
    r = font(FONT_REG, 22)
    d.text((200, 40), "Health Seminar", font=t, fill="#555555")
    d.line([(80, 90), (560, 90)], fill="#999999", width=2)
    lines = [
        "Seminar on unproven herbal remedy.",
        "Claims to cure multiple conditions.",
        "Featured speaker not medically licensed.",
        "Tickets $199 per person, non-refundable.",
    ]
    y = 120
    for line in lines:
        d.text((80, y), line, font=r, fill="black")
        y += 40
    d.text((80, 350), "Editorial review pending - claims unverified", font=font(FONT_REG, 20), fill="#a05a1e")
    return img


# ── 生成 ─────────────────────────────────────────────────────────────

OUT = {
    "datasets/document_classification/sample_invoice.jpg": doc_invoice,
    "datasets/document_classification/sample_contract.jpg": doc_contract,
    "datasets/document_classification/sample_report.jpg": doc_report,
    "datasets/document_classification/sample_letter.jpg": doc_letter,
    "datasets/document_classification/sample_receipt.jpg": doc_receipt,
    "datasets/scene_recognition/indoor_1.jpg": scene_indoor_1,
    "datasets/scene_recognition/indoor_2.jpg": scene_indoor_2,
    "datasets/scene_recognition/outdoor_1.jpg": scene_outdoor_1,
    "datasets/scene_recognition/outdoor_2.jpg": scene_outdoor_2,
    "datasets/scene_recognition/night_1.jpg": scene_night_1,
    "datasets/content_moderation/sample_normal_1.jpg": mod_normal_1,
    "datasets/content_moderation/sample_normal_2.jpg": mod_normal_2,
    "datasets/content_moderation/sample_violation_1.jpg": mod_violation_1,
    "datasets/content_moderation/sample_review_1.jpg": mod_review_1,
}

if __name__ == "__main__":
    for path, fn in OUT.items():
        img = fn()
        img.save(path, quality=90)
        print(f"saved {path} ({img.size[0]}x{img.size[1]})")
    # assets/sample.jpg — 快速开始示例：一张发票图片
    img = doc_invoice()
    img.save("assets/sample.jpg", quality=90)
    print("saved assets/sample.jpg")

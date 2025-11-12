# Create a navy-themed flowchart image with two paths after convolution.
from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 950
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

# Colors
navy = (20, 52, 101)
blue = (36, 97, 183)
light_blue = (206, 226, 255)
mid_blue = (150, 190, 255)
teal = (70, 140, 180)
gray = (90, 90, 90)
accent = (26, 115, 232)
box_fill = (232, 241, 255)

# Fonts (fallback to default)
try:
    title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
    h_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
    text_font = ImageFont.truetype("DejaVuSans.ttf", 26)
    small_font = ImageFont.truetype("DejaVuSans.ttf", 22)
except:
    title_font = h_font = text_font = small_font = ImageFont.load_default()

def rrect(xy, radius=20, outline=navy, fill=box_fill, width=3):
    x0,y0,x1,y1 = xy
    draw.rounded_rectangle(xy, radius=radius, outline=outline, width=width, fill=fill)

def arrow(p0, p1, color=navy, width=4, head=16):
    # draw line with arrowhead
    draw.line([p0, p1], fill=color, width=width)
    # arrowhead
    from math import atan2, cos, sin
    x0,y0 = p0; x1,y1 = p1
    ang = atan2(y1-y0, x1-x0)
    hx = x1 - head*cos(ang)
    hy = y1 - head*sin(ang)
    left = (hx - head*cos(ang - 2.4), hy - head*sin(ang - 2.4))
    right = (hx - head*cos(ang + 2.4), hy - head*sin(ang + 2.4))
    draw.polygon([p1, left, right], fill=color)

def label(text, xy, font=text_font, fill=navy, align="center"):
    x0,y0,x1,y1 = xy
    w = x1-x0; h = y1-y0
    tw, th = draw.textbbox((0,0), text, font=font)[2:4]
    tx = x0 + (w-tw)//2
    ty = y0 + (h-th)//2
    draw.text((tx,ty), text, font=font, fill=fill, align=align)

# Title
draw.text((W//2 - 420, 30), "Uncertainty-Aware CNN: Branching After Convolution", font=title_font, fill=navy)

# Left stack: Input and Encoding
inp = (80, 150, 330, 310)
rrect(inp, fill=(245,250,255))
label("Noisy image\n(H×W, 0–255)", inp, font=text_font)

enc = (360, 150, 640, 310)
rrect(enc)
label("Input encoding", (360,150,640,190), font=h_font)
# U/V subboxes
u = (370, 205, 630, 255)
v = (370, 260, 630, 310)
draw.rectangle(u, outline=blue, width=2, fill=(222,235,255))
draw.rectangle(v, outline=teal, width=2, fill=(222,245,255))
draw.text((u[0]+10,u[1]+4), "U-part (K₁ LSB: −1/0/+1)", font=small_font, fill=navy)
draw.text((v[0]+10,v[1]+4), "V-part (K₂ MSB: balanced −1/0/+1)", font=small_font, fill=navy)

# U-pass and V-pass boxes
upass = (720, 130, 1000, 230)
vpass = (720, 250, 1000, 350)
rrect(upass); label("U-pass (uncertain binary)\nSK ripple add ⟶ carry", upass, font=text_font)
rrect(vpass); label("V-pass (balanced ternary)\nBT adder ⟶ provisional", vpass, font=text_font)

# Arrows from encoding to passes
arrow((640, 200), (720, 180))
arrow((640, 260), (720, 300))

# U' and carry
uprime = (1050, 150, 1145, 200)
carry = (1050, 205, 1185, 250)
rrect(uprime, radius=10); label("U′", uprime, font=text_font)
rrect(carry, radius=10); label("carry", carry, font=small_font)

# V' provisional
vprime = (1050, 280, 1145, 330)
rrect(vprime, radius=10); label("V′*", vprime, font=text_font)

# Recombine
recomb = (1210, 200, 1500, 310)
rrect(recomb, fill=(240,248,255))
label("Recombine  [ U′  |  V′ ]\n(carry injected only when certain)", recomb, font=text_font)

# arrows into recombine
arrow((1145, 175), (1210, 225))
arrow((1185, 230), (1210, 245))
arrow((1145, 305), (1210, 285))

# Branch header
draw.text((280, 370), "Two downstream options:", font=h_font, fill=navy)

# Path A: Resolve uncertainty -> float
a1 = (120, 420, 470, 510)
rrect(a1); label("A) Resolve uncertainty\nDecode to numeric view", a1, font=text_font)
a2 = (120, 545, 470, 625)
rrect(a2); label("Standard pooling / conv\n(float domain)", a2, font=text_font)
a3 = (120, 660, 470, 750)
rrect(a3); label("Classifier head\nDense → Softmax", a3, font=text_font)

# Path B: Stay encoded
b1 = (560, 420, 910, 510)
rrect(b1); label("B) Stay encoded\nTernary/uncertain domain", b1, font=text_font)
b2 = (560, 545, 910, 625)
rrect(b2); label("Ternary pooling / further\nU/V conv blocks", b2, font=text_font)
b3 = (560, 660, 910, 750)
rrect(b3); label("Encoded classifier head\n(quantized logits)", b3, font=text_font)

# Training box for B with ReSTE
train = (980, 545, 1480, 750)
rrect(train, fill=(235,244,255))
label("Training for Path B:\n• ReSTE / STE for backprop\n• Quantization-aware training\n• Optional interval losses", train, font=text_font)

# arrows from recombine to branches
arrow((1355, 310), (330, 420))  # to A
arrow((1355, 310), (735, 420))  # to B

# arrows within branches
arrow((295, 510), (295, 545))
arrow((295, 625), (295, 660))
arrow((735, 510), (735, 545))
arrow((735, 625), (735, 660))
arrow((910, 585), (980, 585))
arrow((910, 705), (980, 705))

# Footer note
draw.text((140, 830), "Choose A for compatibility and interpretability; choose B for end-to-end uncertainty handling with discrete compute.", font=small_font, fill=gray)
print('Presentation completed.')
path = "Branching_After_Convolution_Two_Paths.png"
img.save(path)
path

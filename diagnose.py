import pymupdf

for pdf in [r"D:\New folder\data\Amazon_2019.pdf", r"D:\New folder\data\Amazon_2020.pdf"]:
    print("=" * 20, pdf)
    doc = pymupdf.open(pdf)
    print("pages:", len(doc))
    for pno in [1, 3, 10, 20, 30, 45, 60, 80]:
        if pno > len(doc):
            continue
        page = doc[pno - 1]
        text = page.get_text()
        print(f"\n--- page {pno}: chars={len(text)} images={len(page.get_images())}")
        print(repr(text[:300]))
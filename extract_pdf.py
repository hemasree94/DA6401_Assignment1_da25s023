from PyPDF2 import PdfReader

pdf_path = r'DA6401_Assignment_1_2026 (1).pdf'
reader = PdfReader(pdf_path)

print(f"Total pages: {len(reader.pages)}\n")

for i, page in enumerate(reader.pages):
    print(f"\n--- PAGE {i+1} ---\n")
    text = page.extract_text()
    print(text)

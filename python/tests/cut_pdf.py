import pikepdf

path = "./temp_Dayan_Abbott.pdf"
a, b = 97, 114
diff = 16

with pikepdf.Pdf.open(path) as pdf:
    new_pdf = pikepdf.Pdf.new()
    for i in range(a, b):
        new_pdf.pages.append(pdf.pages[i - 1 + diff])
    new_pdf.save("temp.pdf")

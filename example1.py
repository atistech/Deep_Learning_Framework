import framework.HandwrittenDigitRecognizer as HDR

hdr = HDR.HandwrittenDigitRecognizer()
result = hdr.getResult("example1.png")
print(result)
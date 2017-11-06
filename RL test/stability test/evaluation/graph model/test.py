def convertToBit(a, outsize):
    i = 0
    ret = []
    while i < outsize:
        ret.append(a & 1)
        a = a >> 1
        i += 1
    return ret

print(convertToBit(3,4))

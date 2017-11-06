c = 's'
while True:
    try:
        print(c)
        a = int(c)
        break
    except ValueError:
        print('not a number')
        c = 2
        print('here')
print('here')

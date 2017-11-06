import keyboard
import time

#def wait(T):
#    for i in list(range(T))[::-1]:
#        print(i+1)
#        time.sleep(1)

#wait(T=4)
while True:
    keyboard.press('a')
    keyboard.press('k')
    time.sleep(0.5)
    keyboard.release('a')
    keyboard.release('k')
    time.sleep(0.5)
    #for i in range(20):
    #    keyboard.release('d')
    #keyboard.press_and_release('d')
    #keyboard.press('d')
    #keyboard.press_and_release('w')
    #time.sleep(0.3)
    #keyboard.press_and_release('w')
    #keyboard.release('d')

    #keyboard.press_and_release('j')
    #time.sleep(0.3)
    #keyboard.press_and_release('k')
#while True:
#    keyboard.press('d')
#    keyboard.press_and_release('w')

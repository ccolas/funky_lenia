import mido
import time

port = mido.open_input()
while True:
    for msg in port.iter_pending():
        print(msg)

    print('start pause')
    time.sleep(5)
    print('stop pause')
for /r G:\pppp\ %%i in (*) do @FaceDetection -f haarcascade_frontalface_alt2.xml -m model.bin -t train.txt -i %%i
PAUSE
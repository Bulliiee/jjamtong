from multiprocessing import Process
from web import app_start

#프로세스 실행
process1 = Process(target = app_start.app.run('0.0.0.0', port=8080, debug=True)).start()